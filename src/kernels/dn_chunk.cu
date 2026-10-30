// Chunked (WY) form of the gated delta rule on tensor cores, for prefill chunks.
//
// Per value head, with state S [v][k] and the tokens i of a 64-token chunk (decays a_i, betas
// b_i, gamma_i = a_0 * .. * a_i within the chunk), the recurrence
//     S_i = a_i S_{i-1} (I - b_i k_i k_i^T) + b_i v_i k_i^T,   o_i = S_i q_i
// is S_i = gamma_i S0 + sum_{j<=i} (gamma_i / gamma_j) u_j k_j^T with pseudo-values
//     U = T (diag(b) V - diag(b gamma) K S0^T),  T = (I + L)^-1,
//     L_ij = b_i (gamma_i / gamma_j) k_i.k_j  (j < i)
// so with W = T diag(b gamma) K and U~ = T diag(b) V:
//     U = U~ - W S0^T
//     O = diag(gamma) Q S0^T + (M o Q K^T) U,   M_ij = gamma_i / gamma_j (j <= i)
//     S_C = gamma_C S0 + U^T diag(gamma_C / gamma) K
// dn_chunk_prep: everything independent of S0, for all chunks in parallel (MMA: fp16 operands,
// fp32 accumulation; the triangular inverse T in fp32). dn_chunk_scan: the chunks of one head in
// order, a 64-column v slice per block, the state slice in fp32 MMA accumulators.
#include "../env.h"
#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>

#include "ops.cuh"
#include "gemv.cuh"

namespace {

constexpr int C = 64;    // tokens per chunk
constexpr int DK = 128;  // key / value head dim
constexpr int VS = 64;   // v columns per scan block
constexpr int kThreads = 128;      // scan
constexpr int kPrepThreads = 256;  // prep: 8 warps (one block per SM: latency hiding)

// per (head, chunk) scratch in halves: W, Qh (gamma Q), Kh ((gamma_C / gamma) K), U~ [C][DK]
// and A = M o QK^T [C][C]
constexpr size_t kMat = (size_t)C * DK;
constexpr size_t kPer = 4 * kMat + (size_t)C * C;

__device__ __forceinline__ void ldsm_x4(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                     (unsigned)__cvta_generic_to_shared(smem)),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// fp16 tiles in shared memory, rows of R halves (R = 64 or 128) split in 16-byte chunks whose
// index is XOR-swizzled with (row & 7): ldmatrix row reads are bank-conflict free.
// (rows of fewer than 8 chunks, R = 16 / 32: the XOR uses only as many row bits as there are chunks)
template <int R>
__device__ __forceinline__ __half* at(__half* base, int row, int col) {
    constexpr int SW = R / 8 >= 8 ? 7 : R / 8 - 1;
    return base + row * R + ((((col >> 3) ^ (row & SW)) << 3) | (col & 7));
}
template <int R>
__device__ __forceinline__ const __half* at(const __half* base, int row, int col) {
    constexpr int SW = R / 8 >= 8 ? 7 : R / 8 - 1;
    return base + row * R + ((((col >> 3) ^ (row & SW)) << 3) | (col & 7));
}

// A fragment (16 x 16, row-major source X[row][col]) at rows r0, cols c0
template <int R>
__device__ __forceinline__ void frag_a(uint32_t a[4], const __half* X, int r0, int c0, int lane) {
    ldsm_x4(a, at<R>(X, r0 + (lane & 15), c0 + ((lane >> 4) << 3)));
}
// A fragment of X^T (X stored [col][row]): rows r0.. of X^T = columns of X
template <int R>
__device__ __forceinline__ void frag_a_t(uint32_t a[4], const __half* X, int r0, int c0, int lane) {
    // matrices: (rows r0..7, k c0..7), (rows r0+8.., k c0..), (rows r0.., k c0+8..), (r0+8, c0+8)
    const int mi = lane >> 3, j = lane & 7;
    const int k = c0 + ((mi >> 1) << 3) + j, r = r0 + ((mi & 1) << 3);
    ldsm_x4_t(a, at<R>(X, k, r));
}
// B fragments for two n8 tiles (n0, n0 + 8), k16 at k0, from Y stored [n][k] (B = Y^T)
template <int R>
__device__ __forceinline__ void frag_b_nk(uint32_t b[4], const __half* Y, int n0, int k0, int lane) {
    const int mi = lane >> 3, j = lane & 7;
    ldsm_x4(b, at<R>(Y, n0 + ((mi >> 1) << 3) + j, k0 + ((mi & 1) << 3)));
    // b[0], b[1]: tile n0 (k0.., k0+8..); b[2], b[3]: tile n0 + 8
}
// B fragments for two n8 tiles from Y stored [k][n]
template <int R>
__device__ __forceinline__ void frag_b_kn(uint32_t b[4], const __half* Y, int n0, int k0, int lane) {
    const int mi = lane >> 3, j = lane & 7;
    ldsm_x4_t(b, at<R>(Y, k0 + ((mi & 1) << 3) + j, n0 + ((mi >> 1) << 3)));
}

// ---------------------------------------------------------------- prep
constexpr int LS = C + 1;  // fp32 matrix row stride
constexpr size_t kPrepSmem =
    (size_t)4 * C * DK * 2 + (size_t)C * C * 2 + (size_t)C * LS * 4 + 3 * C * 4;

__device__ __forceinline__ float silu_c(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_c(float x) { return 1.f / (1.f + __expf(-x)); }

// H16: q, k (qk) and the v part of cv are fp16 (dn_convqk_win_kernel with QE_DN_H16): half the input bytes.
// FUSED (ci): the prep computes the front end itself from the in_proj rows - conv + SiLU, q / k
// L2 norm (warp reduction), decay / beta from the ab projection - no qk / gb / cv buffers.
template <bool H16 = false, bool FUSED = false>
__global__ void __launch_bounds__(kPrepThreads) dn_chunk_prep_kernel(
    const float* __restrict__ cv, int ch, DeltaShape s, const float* __restrict__ qk,
    const float* __restrict__ gb, int m, __half* __restrict__ scr, float* __restrict__ gcs, DnConvIn ci) {
    extern __shared__ __align__(128) unsigned char smraw[];
    __half* Kf = reinterpret_cast<__half*>(smraw);  // k                    [C][DK]
    __half* Qf = Kf + C * DK;                         // q                    [C][DK]
    __half* K2 = Qf + C * DK;                         // b gamma k            [C][DK]
    __half* V2 = K2 + C * DK;                         // b v                  [C][DK]
    __half* Th = V2 + C * DK;                         // T                    [C][C]
    float* Lm = reinterpret_cast<float*>(Th + C * C); // L, then T in fp32    [C][LS]
    float* g = Lm + C * LS;                           // cumulative log decay
    float* be = g + C;
    float* gam = be + C;
    const int h = blockIdx.x, c = blockIdx.y, nch = gridDim.y, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, gq = lane >> 2, tq = lane & 3;
    const int kh = s.v2k[h];
    const int t0 = c * C, n = min(C, m - t0);
    const size_t qk_row = (size_t)2 * s.nk * DK;
    __half* out = scr + ((size_t)h * nch + c) * kPer;
    __half* Wg = out;
    __half* Qg = Wg + kMat;
    __half* Kg = Qg + kMat;
    __half* Ug = Kg + kMat;
    __half* Ag = Ug + kMat;

    if (t < C) {  // padded tokens: decay 1, beta 0 (no effect on the state)
        const bool ok = t < n;
        float dec = 1.f, bet = 0.f;
        if (ok) {
            if constexpr (FUSED) {  // as dn_qk_kernel
                const int ldab = 2 * s.ab_pad;
                float bb = 0.f, aa = ci.dt_bias[h];
                for (int sp = 0; sp < s.ab_splits; sp++) {
                    const float* abs = ci.ab + (size_t)sp * s.ab_rows * ldab + (size_t)(t0 + t) * ldab;
                    bb += abs[h];
                    aa += abs[s.ab_pad + h];
                }
                const float sp_a = aa > 20.f ? aa : log1pf(__expf(aa));
                dec = __expf(ci.A[h] * sp_a);
                bet = sigmoid_c(bb);
            } else {
                dec = gb[(size_t)(t0 + t) * 2 * s.nv + h];
                bet = gb[(size_t)(t0 + t) * 2 * s.nv + s.nv + h];
            }
        }
        // decays can underflow to 0 (a token that wipes the state): log(0) = -inf would make
        // gamma_i / gamma_j = exp(-inf + inf) = NaN; exp(-69) per token is still exactly 0 below
        g[t] = ok ? __logf(fmaxf(dec, 1e-30f)) : 0.f;
        be[t] = ok ? bet : 0.f;
    }
    __syncthreads();
    if (t < 32) {  // inclusive prefix sum of the 64 log decays (two per lane)
        const float a = g[2 * t], b = a + g[2 * t + 1];
        float x = b;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, x, o);
            if (t >= o) x += y;
        }
        g[2 * t] = x - b + a;
        g[2 * t + 1] = x;
    }
    __syncthreads();
    if (t < C) gam[t] = __expf(g[t]);
    __syncthreads();
    const float gl = g[C - 1];
    // inputs -> smem (fp16), Qh / Kh -> scratch
    float wq[3][4][4], xw[3][4][3];  // FUSED: conv taps and the 3-token window of 12 channels per lane
    int chn[3] = {0, 0, 0};
    const int p0 = FUSED ? ci.pos[0] : 0;
    auto hv = [&](int c2, int q) {  // conv input of channel c2 at absolute position q (dn_conv_par_kernel)
        if (q < 0) return 0.f;
        if (q >= p0) return __bfloat162float(__float2bfloat16(ci.proj[(size_t)(q - p0) * ci.ld + c2]));
        return __bfloat162float(ci.hist[(size_t)(q % kConvRing) * ch + c2]);
    };
    if constexpr (FUSED) {  // warp w: tokens 8 w .. 8 w + 7 in order, lane: k = 4 lane .. 4 lane + 3
        const int k = lane * 4;
        chn[0] = kh * DK + k;
        chn[1] = s.nk * DK + kh * DK + k;
        chn[2] = 2 * s.nk * DK + h * DK + k;
        const int pa = p0 + t0 + warp * 8;
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c2 = chn[a] + j;
#pragma unroll
                for (int tp = 0; tp < 4; tp++) wq[a][j][tp] = __bfloat162float(ci.w[c2 * 4 + tp]);
                xw[a][j][0] = hv(c2, pa - 3);
                xw[a][j][1] = hv(c2, pa - 2);
                xw[a][j][2] = hv(c2, pa - 1);
            }
    }
    for (int e = t; e < C * DK / 4; e += kPrepThreads) {
        const int i = FUSED ? warp * 8 + (e - t) / kPrepThreads : e / (DK / 4), k = FUSED ? lane * 4 : (e % (DK / 4)) * 4;
        float4 q4 = make_float4(0.f, 0.f, 0.f, 0.f), k4 = q4, v4 = q4;
        if constexpr (FUSED) {
            float y[3][4];
#pragma unroll
            for (int a = 0; a < 3; a++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float x0 = i < n ? hv(chn[a] + j, p0 + t0 + i) : 0.f;
                    y[a][j] = silu_c(wq[a][j][0] * xw[a][j][0] + wq[a][j][1] * xw[a][j][1] +
                                     wq[a][j][2] * xw[a][j][2] + wq[a][j][3] * x0);
                    xw[a][j][0] = xw[a][j][1];
                    xw[a][j][1] = xw[a][j][2];
                    xw[a][j][2] = x0;
                }
            float sq = y[0][0] * y[0][0] + y[0][1] * y[0][1] + y[0][2] * y[0][2] + y[0][3] * y[0][3];
            float sk = y[1][0] * y[1][0] + y[1][1] * y[1][1] + y[1][2] * y[1][2] + y[1][3] * y[1][3];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                sq += __shfl_xor_sync(0xffffffffu, sq, o);
                sk += __shfl_xor_sync(0xffffffffu, sk, o);
            }
            if (i < n) {
                const float rq = rsqrtf(sq + 1e-6f) * 0.08838834764831845f, rk = rsqrtf(sk + 1e-6f);
                q4 = make_float4(y[0][0] * rq, y[0][1] * rq, y[0][2] * rq, y[0][3] * rq);
                k4 = make_float4(y[1][0] * rk, y[1][1] * rk, y[1][2] * rk, y[1][3] * rk);
                v4 = make_float4(y[2][0], y[2][1], y[2][2], y[2][3]);
            }
        } else if (i < n) {
            if constexpr (H16) {
                auto ld4 = [](const __half* p) {
                    const uint2 u = *reinterpret_cast<const uint2*>(p);
                    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                    return make_float4(a.x, a.y, b.x, b.y);
                };
                const __half* r = reinterpret_cast<const __half*>(qk) + (size_t)(t0 + i) * qk_row;
                q4 = ld4(r + kh * DK + k);
                k4 = ld4(r + s.nk * DK + kh * DK + k);
                v4 = ld4(reinterpret_cast<const __half*>(cv) + (size_t)(t0 + i) * ch + 2 * s.nk * DK + h * DK + k);
            } else {
                const float* r = qk + (size_t)(t0 + i) * qk_row;
                q4 = *reinterpret_cast<const float4*>(r + kh * DK + k);
                k4 = *reinterpret_cast<const float4*>(r + s.nk * DK + kh * DK + k);
                v4 = *reinterpret_cast<const float4*>(cv + (size_t)(t0 + i) * ch + 2 * s.nk * DK + h * DK + k);
            }
        }
        const float gi = gam[i], kc = __expf(gl - g[i]), bg = be[i] * gi, bi = be[i];
        auto st2 = [&](__half* X, float a, float b, float c2, float d) {
            *reinterpret_cast<uint2*>(at<DK>(X, i, k)) = make_uint2(h2(a, b), h2(c2, d));
        };
        st2(Kf, k4.x, k4.y, k4.z, k4.w);
        st2(Qf, q4.x, q4.y, q4.z, q4.w);
        st2(K2, bg * k4.x, bg * k4.y, bg * k4.z, bg * k4.w);
        st2(V2, bi * v4.x, bi * v4.y, bi * v4.z, bi * v4.w);
        *reinterpret_cast<uint2*>(Qg + i * DK + k) =
            make_uint2(h2(gi * q4.x, gi * q4.y), h2(gi * q4.z, gi * q4.w));
        *reinterpret_cast<uint2*>(Kg + i * DK + k) =
            make_uint2(h2(kc * k4.x, kc * k4.y), h2(kc * k4.z, kc * k4.w));
    }
    __syncthreads();
    // K K^T (warps 0-3: L, fp32 smem) and Q K^T (warps 4-7: A, fp16 scratch), 16 rows each
    {
        const int r0 = (warp & 3) * 16;
        const bool isq = warp >= 4;
        float kk[8][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < DK; k0 += 16) {
            uint32_t a[4];
            frag_a<DK>(a, isq ? Qf : Kf, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < 4; np++) {
                uint32_t b[4];
                frag_b_nk<DK>(b, Kf, np * 16, k0, lane);
                mma(kk[np * 2], a, b[0], b[1]);
                mma(kk[np * 2 + 1], a, b[2], b[3]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + gq + hf * 8, j = nt * 8 + 2 * tq;
                const float d0 = j <= i ? __expf(g[i] - g[j]) : 0.f;
                const float d1 = j + 1 <= i ? __expf(g[i] - g[j + 1]) : 0.f;
                const float x0 = kk[nt][hf * 2], x1 = kk[nt][hf * 2 + 1];
                if (isq) {
                    *reinterpret_cast<uint32_t*>(Ag + i * C + j) = h2(d0 * x0, d1 * x1);
                } else {
                    Lm[i * LS + j] = j < i ? be[i] * d0 * x0 : 0.f;
                    Lm[i * LS + j + 1] = j + 1 < i ? be[i] * d1 * x1 : 0.f;
                }
            }
    }
    __syncthreads();
    // T = (I + L)^-1 (unit lower triangular), blockwise with 16 x 16 blocks: Tm reuses Qf's
    // space, Kf's holds the per-warp products. Diagonal blocks by forward substitution within
    // a warp; then off-diagonal blocks by distance d: T_IJ = -T_II sum_{K=J}^{I-1} L_IK T_KJ.
    float* Tm = reinterpret_cast<float*>(Qf);  // [C][C]
    float* Xs = reinterpret_cast<float*>(Kf);  // [warp][16][16]
    if (warp < 4) {
        // lane jj (< 16) owns column b0 + jj of the diagonal block, kept in registers
        const int b0 = warp * 16, jj = lane & 15;
        float tc[16];
#pragma unroll
        for (int ii = 0; ii < 16; ii++) {
            float acc = 0.f;
#pragma unroll
            for (int l = 0; l < ii; l++)  // l >= jj only (tc[l] = 0 above the diagonal)
                acc += Lm[(b0 + ii) * LS + b0 + l] * tc[l];
            tc[ii] = jj == ii ? 1.f : (jj < ii ? -acc : 0.f);
        }
        for (int e = lane; e < 16 * C; e += 32) {  // this warp's rows of Tm: zero
            const int i = b0 + e / C, j = e % C;
            Tm[i * C + j] = 0.f;
        }
        __syncwarp();
        if (lane < 16)
#pragma unroll
            for (int ii = 0; ii < 16; ii++) Tm[(b0 + ii) * C + b0 + jj] = tc[ii];
    }
    __syncthreads();
    for (int d = 1; d < 4; d++) {
        if (warp < 4 - d) {
            const int J = warp, I = warp + d;
            const int r = lane & 15, c0 = (lane >> 4) * 8;
            float x[8] = {};
            for (int K = J; K < I; K++)
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    const float l = Lm[(16 * I + r) * LS + 16 * K + kk];
                    const float4 t0 = *reinterpret_cast<const float4*>(Tm + (16 * K + kk) * C + 16 * J + c0);
                    const float4 t1 = *reinterpret_cast<const float4*>(Tm + (16 * K + kk) * C + 16 * J + c0 + 4);
                    x[0] += l * t0.x; x[1] += l * t0.y; x[2] += l * t0.z; x[3] += l * t0.w;
                    x[4] += l * t1.x; x[5] += l * t1.y; x[6] += l * t1.z; x[7] += l * t1.w;
                }
            float* X = Xs + warp * 256;
            *reinterpret_cast<float4*>(X + r * 16 + c0) = make_float4(x[0], x[1], x[2], x[3]);
            *reinterpret_cast<float4*>(X + r * 16 + c0 + 4) = make_float4(x[4], x[5], x[6], x[7]);
            __syncwarp();
            float y[8] = {};
#pragma unroll
            for (int q2 = 0; q2 < 16; q2++) {  // T_II is lower triangular (zeros above)
                const float tv = Tm[(16 * I + r) * C + 16 * I + q2];
                const float4 x0 = *reinterpret_cast<const float4*>(X + q2 * 16 + c0);
                const float4 x1 = *reinterpret_cast<const float4*>(X + q2 * 16 + c0 + 4);
                y[0] += tv * x0.x; y[1] += tv * x0.y; y[2] += tv * x0.z; y[3] += tv * x0.w;
                y[4] += tv * x1.x; y[5] += tv * x1.y; y[6] += tv * x1.z; y[7] += tv * x1.w;
            }
            *reinterpret_cast<float4*>(Tm + (16 * I + r) * C + 16 * J + c0) = make_float4(-y[0], -y[1], -y[2], -y[3]);
            *reinterpret_cast<float4*>(Tm + (16 * I + r) * C + 16 * J + c0 + 4) = make_float4(-y[4], -y[5], -y[6], -y[7]);
        }
        __syncthreads();
    }
    for (int e = t; e < C * C; e += kPrepThreads) {
        const int i = e / C, j = e % C;
        *at<C>(Th, i, j) = __float2half_rn(Tm[e]);
    }
    __syncthreads();
    // W = T K2 (warps 0-3), U~ = T V2 (warps 4-7): 16 rows, 128 columns each
    {
        const int r0 = (warp & 3) * 16;
        uint32_t a[4][4];
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) frag_a<C>(a[k0 / 16], Th, r0, k0, lane);
        {
            const int which = warp >> 2;
            const __half* B = which ? V2 : K2;
            __half* O = which ? Ug : Wg;
#pragma unroll
            for (int np = 0; np < DK / 16; np++) {
                float acc[2][4] = {};
#pragma unroll
                for (int k0 = 0; k0 < C; k0 += 16) {
                    uint32_t b[4];
                    frag_b_kn<DK>(b, B, np * 16, k0, lane);
                    mma(acc[0], a[k0 / 16], b[0], b[1]);
                    mma(acc[1], a[k0 / 16], b[2], b[3]);
                }
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const int col = np * 16 + x * 8 + 2 * tq;
                    *reinterpret_cast<uint32_t*>(O + (r0 + gq) * DK + col) = h2(acc[x][0], acc[x][1]);
                    *reinterpret_cast<uint32_t*>(O + (r0 + gq + 8) * DK + col) = h2(acc[x][2], acc[x][3]);
                }
            }
        }
    }
    if (t == 0) gcs[(size_t)h * nch + c] = __expf(gl);
}

// ---------------------------------------------------------------- scan
// smem: W, Qh, Kh [C][DK], A [C][C], U~ slice [C][VS], S (fp16) [VS][DK], U (fp16) [C][VS]
constexpr size_t kScanSmem = ((size_t)3 * C * DK + C * C + C * VS + VS * DK + C * VS) * 2;

__global__ void __launch_bounds__(kThreads, 1) dn_chunk_scan_kernel(
    DeltaShape s, const __half* __restrict__ scr, const float* __restrict__ gcs, int nch, int m,
    float* __restrict__ S, size_t slot_floats, const int* __restrict__ curp, int slot_off,
    float* __restrict__ o) {
    extern __shared__ __align__(128) unsigned char smraw[];
    __half* Ws = reinterpret_cast<__half*>(smraw);
    __half* Qs = Ws + C * DK;
    __half* Ks = Qs + C * DK;
    __half* As = Ks + C * DK;
    __half* Ut = As + C * C;   // [C][VS]
    __half* Ss = Ut + C * VS;  // [VS][DK]
    __half* Us = Ss + VS * DK; // [C][VS]
    const int h = blockIdx.x, v0 = blockIdx.y * VS, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, gq = lane >> 2, tq = lane & 3;
    const int cur = (*curp + slot_off) % kStateSlots;
    const int r0 = warp * 16;  // this warp's v rows (state) / token rows (U, O)
    // state slice S[v0 + r0 .. +16][0 .. 128) as m16n8 accumulators
    float st[DK / 8][4];
    {
        const float* sp = S + (size_t)cur * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                st[nt][e] = sp[(size_t)(r0 + gq + ((e >> 1) << 3)) * DK + nt * 8 + 2 * tq + (e & 1)];
    }
    for (int c = 0; c < nch; c++) {
        const __half* src = scr + ((size_t)h * nch + c) * kPer;
        // load this chunk's operands (swizzled)
        for (int e = t; e < C * DK / 8; e += kThreads) {
            const int i = e / (DK / 8), k = (e % (DK / 8)) * 8;
            cp_async16(at<DK>(Ws, i, k), src + i * DK + k);
            cp_async16(at<DK>(Qs, i, k), src + kMat + i * DK + k);
            cp_async16(at<DK>(Ks, i, k), src + 2 * kMat + i * DK + k);
        }
        for (int e = t; e < C * VS / 8; e += kThreads) {
            const int i = e / (VS / 8), k = (e % (VS / 8)) * 8;
            cp_async16(at<VS>(Ut, i, k), src + 3 * kMat + i * DK + v0 + k);
        }
        for (int e = t; e < C * C / 8; e += kThreads) {
            const int i = e / (C / 8), k = (e % (C / 8)) * 8;
            cp_async16(at<C>(As, i, k), src + 4 * kMat + i * C + k);
        }
        cp_commit();
        // state -> fp16 smem (B operand of W S^T and Qh S^T)
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++) {
            const int col = nt * 8 + 2 * tq;
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, r0 + gq, col)) = h2(st[nt][0], st[nt][1]);
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, r0 + gq + 8, col)) = h2(st[nt][2], st[nt][3]);
        }
        cp_wait0();
        __syncthreads();
        // P = W S^T and O = Qh S^T for this warp's 16 token rows, 64 v columns
        float p[VS / 8][4] = {}, ov[VS / 8][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < DK; k0 += 16) {
            uint32_t aw[4], aq[4];
            frag_a<DK>(aw, Ws, r0, k0, lane);
            frag_a<DK>(aq, Qs, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VS / 16; np++) {
                uint32_t b[4];
                frag_b_nk<DK>(b, Ss, np * 16, k0, lane);
                mma(p[np * 2], aw, b[0], b[1]);
                mma(p[np * 2 + 1], aw, b[2], b[3]);
                mma(ov[np * 2], aq, b[0], b[1]);
                mma(ov[np * 2 + 1], aq, b[2], b[3]);
            }
        }
        // U = U~ - P -> fp16 smem
#pragma unroll
        for (int nt = 0; nt < VS / 8; nt++) {
            const int col = nt * 8 + 2 * tq;
            const __half2 u0 = *reinterpret_cast<const __half2*>(at<VS>(Ut, r0 + gq, col));
            const __half2 u1 = *reinterpret_cast<const __half2*>(at<VS>(Ut, r0 + gq + 8, col));
            const float2 f0 = __half22float2(u0), f1 = __half22float2(u1);
            *reinterpret_cast<uint32_t*>(at<VS>(Us, r0 + gq, col)) = h2(f0.x - p[nt][0], f0.y - p[nt][1]);
            *reinterpret_cast<uint32_t*>(at<VS>(Us, r0 + gq + 8, col)) = h2(f1.x - p[nt][2], f1.y - p[nt][3]);
        }
        __syncthreads();
        // O += A U (A lower triangular: only k16 blocks <= this warp's rows)
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            if (k0 > r0) break;
            uint32_t a[4];
            frag_a<C>(a, As, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VS / 16; np++) {
                uint32_t b[4];
                frag_b_kn<VS>(b, Us, np * 16, k0, lane);
                mma(ov[np * 2], a, b[0], b[1]);
                mma(ov[np * 2 + 1], a, b[2], b[3]);
            }
        }
        const int t0 = c * C;
#pragma unroll
        for (int nt = 0; nt < VS / 8; nt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + gq + hf * 8;
                if (t0 + i < m)
                    *reinterpret_cast<float2*>(o + ((size_t)(t0 + i) * s.nv + h) * DK + v0 + nt * 8 + 2 * tq) =
                        make_float2(ov[nt][hf * 2], ov[nt][hf * 2 + 1]);
            }
        // S = gamma_C S + U^T Kh (this warp's 16 v rows, all 128 k)
        const float gc = gcs[(size_t)h * nch + c];
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) st[nt][e] *= gc;
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            uint32_t a[4];
            frag_a_t<VS>(a, Us, r0, k0, lane);  // (U^T)[v][tok] from Us[tok][v]
#pragma unroll
            for (int np = 0; np < DK / 16; np++) {
                uint32_t b[4];
                frag_b_kn<DK>(b, Ks, np * 16, k0, lane);
                mma(st[np * 2], a, b[0], b[1]);
                mma(st[np * 2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();  // smem is rewritten by the next chunk
    }
    float* dp = S + (size_t)((cur + m) % kStateSlots) * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
    for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++)
            *reinterpret_cast<float2*>(dp + (size_t)(r0 + gq + hf * 8) * DK + nt * 8 + 2 * tq) =
                make_float2(st[nt][hf * 2], st[nt][hf * 2 + 1]);
}

__global__ void __launch_bounds__(kThreads, 1) dn_chunk_scan2_kernel(
    DeltaShape s, const __half* __restrict__ scr, const float* __restrict__ gcs, int nch, int m,
    float* __restrict__ S, size_t slot_floats, const int* __restrict__ curp, int slot_off,
    float* __restrict__ o) {
    extern __shared__ __align__(128) unsigned char smraw[];
    __half* Ws = reinterpret_cast<__half*>(smraw);
    __half* Qs = Ws + C * DK;
    __half* Ks = Qs + C * DK;
    __half* As = Ks + C * DK;
    __half* Ut = As + C * C;   // [C][VS]
    __half* Ss = Ut + C * VS;  // [VS][DK]
    __half* Us = Ss + VS * DK; // [C][VS]
    const int h = blockIdx.x, v0 = blockIdx.y * VS, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, gq = lane >> 2, tq = lane & 3;
    const int cur = (*curp + slot_off) % kStateSlots;
    const int r0 = warp * 16;  // this warp's v rows (state) / token rows (U, O)
    // state slice S[v0 + r0 .. +16][0 .. 128) as m16n8 accumulators
    float st[DK / 8][4];
    {
        const float* sp = S + (size_t)cur * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                st[nt][e] = sp[(size_t)(r0 + gq + ((e >> 1) << 3)) * DK + nt * 8 + 2 * tq + (e & 1)];
    }
    // rolling prefetch: each operand of chunk c + 1 is loaded into its buffer as soon as chunk c
    // is done with it (W, Qh after phase 1; U~, A after phase 2; Kh after the state update), so
    // the loads overlap the MMAs instead of stalling every chunk on L2 / DRAM latency.
    // cp.async groups pending at the top of iteration c (issue order): WQ_c, UA_c, K_c.
    auto ld_wq = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * DK / 8; e += kThreads) {
                const int i = e / (DK / 8), k = (e % (DK / 8)) * 8;
                cp_async16(at<DK>(Ws, i, k), src + i * DK + k);
                cp_async16(at<DK>(Qs, i, k), src + kMat + i * DK + k);
            }
        }
        cp_commit();
    };
    auto ld_ua = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * VS / 8; e += kThreads) {
                const int i = e / (VS / 8), k = (e % (VS / 8)) * 8;
                cp_async16(at<VS>(Ut, i, k), src + 3 * kMat + i * DK + v0 + k);
            }
            for (int e = t; e < C * C / 8; e += kThreads) {
                const int i = e / (C / 8), k = (e % (C / 8)) * 8;
                cp_async16(at<C>(As, i, k), src + 4 * kMat + i * C + k);
            }
        }
        cp_commit();
    };
    auto ld_k = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * DK / 8; e += kThreads) {
                const int i = e / (DK / 8), k = (e % (DK / 8)) * 8;
                cp_async16(at<DK>(Ks, i, k), src + 2 * kMat + i * DK + k);
            }
        }
        cp_commit();
    };
    ld_wq(0);
    ld_ua(0);
    ld_k(0);
    for (int c = 0; c < nch; c++) {
        // state -> fp16 smem (B operand of W S^T and Qh S^T)
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++) {
            const int col = nt * 8 + 2 * tq;
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, r0 + gq, col)) = h2(st[nt][0], st[nt][1]);
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, r0 + gq + 8, col)) = h2(st[nt][2], st[nt][3]);
        }
        cp_wait<2>();
        __syncthreads();
        // P = W S^T and O = Qh S^T for this warp's 16 token rows, 64 v columns
        float p[VS / 8][4] = {}, ov[VS / 8][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < DK; k0 += 16) {
            uint32_t aw[4], aq[4];
            frag_a<DK>(aw, Ws, r0, k0, lane);
            frag_a<DK>(aq, Qs, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VS / 16; np++) {
                uint32_t b[4];
                frag_b_nk<DK>(b, Ss, np * 16, k0, lane);
                mma(p[np * 2], aw, b[0], b[1]);
                mma(p[np * 2 + 1], aw, b[2], b[3]);
                mma(ov[np * 2], aq, b[0], b[1]);
                mma(ov[np * 2 + 1], aq, b[2], b[3]);
            }
        }
        __syncthreads();  // all warps are done with Ws / Qs
        ld_wq(c + 1);
        cp_wait<2>();     // U~_c, A_c landed
        __syncthreads();
        // U = U~ - P -> fp16 smem
#pragma unroll
        for (int nt = 0; nt < VS / 8; nt++) {
            const int col = nt * 8 + 2 * tq;
            const __half2 u0 = *reinterpret_cast<const __half2*>(at<VS>(Ut, r0 + gq, col));
            const __half2 u1 = *reinterpret_cast<const __half2*>(at<VS>(Ut, r0 + gq + 8, col));
            const float2 f0 = __half22float2(u0), f1 = __half22float2(u1);
            *reinterpret_cast<uint32_t*>(at<VS>(Us, r0 + gq, col)) = h2(f0.x - p[nt][0], f0.y - p[nt][1]);
            *reinterpret_cast<uint32_t*>(at<VS>(Us, r0 + gq + 8, col)) = h2(f1.x - p[nt][2], f1.y - p[nt][3]);
        }
        __syncthreads();
        // O += A U (A lower triangular: only k16 blocks <= this warp's rows)
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            if (k0 > r0) break;
            uint32_t a[4];
            frag_a<C>(a, As, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VS / 16; np++) {
                uint32_t b[4];
                frag_b_kn<VS>(b, Us, np * 16, k0, lane);
                mma(ov[np * 2], a, b[0], b[1]);
                mma(ov[np * 2 + 1], a, b[2], b[3]);
            }
        }
        const int t0 = c * C;
#pragma unroll
        for (int nt = 0; nt < VS / 8; nt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + gq + hf * 8;
                if (t0 + i < m)
                    *reinterpret_cast<float2*>(o + ((size_t)(t0 + i) * s.nv + h) * DK + v0 + nt * 8 + 2 * tq) =
                        make_float2(ov[nt][hf * 2], ov[nt][hf * 2 + 1]);
            }
        __syncthreads();  // all warps are done with Ut / As
        ld_ua(c + 1);
        cp_wait<2>();     // Kh_c landed
        __syncthreads();
        // S = gamma_C S + U^T Kh (this warp's 16 v rows, all 128 k)
        const float gc = gcs[(size_t)h * nch + c];
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) st[nt][e] *= gc;
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            uint32_t a[4];
            frag_a_t<VS>(a, Us, r0, k0, lane);  // (U^T)[v][tok] from Us[tok][v]
#pragma unroll
            for (int np = 0; np < DK / 16; np++) {
                uint32_t b[4];
                frag_b_kn<DK>(b, Ks, np * 16, k0, lane);
                mma(st[np * 2], a, b[0], b[1]);
                mma(st[np * 2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();  // all warps are done with Ks / Us
        ld_k(c + 1);
    }
    cp_wait<0>();
    float* dp = S + (size_t)((cur + m) % kStateSlots) * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
    for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++)
            *reinterpret_cast<float2*>(dp + (size_t)(r0 + gq + hf * 8) * DK + nt * 8 + 2 * tq) =
                make_float2(st[nt][hf * 2], st[nt][hf * 2 + 1]);
}

// Scan with a VSB-column v slice per block (VSB = 32: 2x the blocks of the 64-column scan, each
// warp's dependent MMA chain per chunk half as long). Warps: token rows r0 = 16 warp for W S^T,
// Qh S^T and A U; the state slice [VSB][128] is split into RG = VSB / 16 row groups x KS = 4 / RG
// k ranges of KW columns. Same arithmetic per output element as dn_chunk_scan_kernel.
template <int VSB>
constexpr size_t scan_smem() { return ((size_t)3 * C * DK + C * C + C * VSB + VSB * DK + C * VSB) * 2; }
template <int VSB, bool PF>
__global__ void __launch_bounds__(kThreads, 1) dn_chunk_scanv_kernel(
    DeltaShape s, const __half* __restrict__ scr, const float* __restrict__ gcs, int nch, int m,
    float* __restrict__ S, size_t slot_floats, const int* __restrict__ curp, int slot_off,
    float* __restrict__ o) {
    constexpr int RG = VSB / 16, KS = 4 / RG, KW = DK / KS;
    extern __shared__ __align__(128) unsigned char smraw[];
    __half* Ws = reinterpret_cast<__half*>(smraw);
    __half* Qs = Ws + C * DK;
    __half* Ks = Qs + C * DK;
    __half* As = Ks + C * DK;
    __half* Ut = As + C * C;    // [C][VSB]
    __half* Ss = Ut + C * VSB;  // [VSB][DK]
    __half* Us = Ss + VSB * DK; // [C][VSB]
    const int h = blockIdx.x, v0 = blockIdx.y * VSB, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, gq = lane >> 2, tq = lane & 3;
    const int cur = (*curp + slot_off) % kStateSlots;
    const int r0 = warp * 16;                               // token rows (P, O)
    const int sr0 = (warp % RG) * 16, kb = (warp / RG) * KW;  // state rows (v) / k columns
    float st[KW / 8][4];
    {
        const float* sp = S + (size_t)cur * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
        for (int nt = 0; nt < KW / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                st[nt][e] = sp[(size_t)(sr0 + gq + ((e >> 1) << 3)) * DK + kb + nt * 8 + 2 * tq + (e & 1)];
    }
    // PF: rolling prefetch, each operand of chunk c + 1 is loaded as soon as chunk c is done with
    // it (W, Qh after W S^T / Qh S^T; U~, A after A U; Kh after the state update). cp.async
    // groups pending at the top of iteration c (issue order): WQ_c, UA_c, K_c.
    auto ld_wq = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * DK / 8; e += kThreads) {
                const int i = e / (DK / 8), k = (e % (DK / 8)) * 8;
                cp_async16(at<DK>(Ws, i, k), src + i * DK + k);
                cp_async16(at<DK>(Qs, i, k), src + kMat + i * DK + k);
            }
        }
        cp_commit();
    };
    auto ld_ua = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * VSB / 8; e += kThreads) {
                const int i = e / (VSB / 8), k = (e % (VSB / 8)) * 8;
                cp_async16(at<VSB>(Ut, i, k), src + 3 * kMat + i * DK + v0 + k);
            }
            for (int e = t; e < C * C / 8; e += kThreads) {
                const int i = e / (C / 8), k = (e % (C / 8)) * 8;
                cp_async16(at<C>(As, i, k), src + 4 * kMat + i * C + k);
            }
        }
        cp_commit();
    };
    auto ld_k = [&](int c) {
        if (c < nch) {
            const __half* src = scr + ((size_t)h * nch + c) * kPer;
            for (int e = t; e < C * DK / 8; e += kThreads) {
                const int i = e / (DK / 8), k = (e % (DK / 8)) * 8;
                cp_async16(at<DK>(Ks, i, k), src + 2 * kMat + i * DK + k);
            }
        }
        cp_commit();
    };
    if constexpr (PF) { ld_wq(0); ld_ua(0); ld_k(0); }
    for (int c = 0; c < nch; c++) {
        if constexpr (!PF) { ld_wq(c); ld_ua(c); ld_k(c); }
#pragma unroll
        for (int nt = 0; nt < KW / 8; nt++) {
            const int col = kb + nt * 8 + 2 * tq;
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, sr0 + gq, col)) = h2(st[nt][0], st[nt][1]);
            *reinterpret_cast<uint32_t*>(at<DK>(Ss, sr0 + gq + 8, col)) = h2(st[nt][2], st[nt][3]);
        }
        if constexpr (PF) cp_wait<2>();
        else cp_wait0();
        __syncthreads();
        float p[VSB / 8][4] = {}, ov[VSB / 8][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < DK; k0 += 16) {
            uint32_t aw[4], aq[4];
            frag_a<DK>(aw, Ws, r0, k0, lane);
            frag_a<DK>(aq, Qs, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VSB / 16; np++) {
                uint32_t b[4];
                frag_b_nk<DK>(b, Ss, np * 16, k0, lane);
                mma(p[np * 2], aw, b[0], b[1]);
                mma(p[np * 2 + 1], aw, b[2], b[3]);
                mma(ov[np * 2], aq, b[0], b[1]);
                mma(ov[np * 2 + 1], aq, b[2], b[3]);
            }
        }
        if constexpr (PF) {
            __syncthreads();  // all warps are done with Ws / Qs
            ld_wq(c + 1);
            cp_wait<2>();     // U~_c, A_c
            __syncthreads();
        }
#pragma unroll
        for (int nt = 0; nt < VSB / 8; nt++) {
            const int col = nt * 8 + 2 * tq;
            const __half2 u0 = *reinterpret_cast<const __half2*>(at<VSB>(Ut, r0 + gq, col));
            const __half2 u1 = *reinterpret_cast<const __half2*>(at<VSB>(Ut, r0 + gq + 8, col));
            const float2 f0 = __half22float2(u0), f1 = __half22float2(u1);
            *reinterpret_cast<uint32_t*>(at<VSB>(Us, r0 + gq, col)) = h2(f0.x - p[nt][0], f0.y - p[nt][1]);
            *reinterpret_cast<uint32_t*>(at<VSB>(Us, r0 + gq + 8, col)) = h2(f1.x - p[nt][2], f1.y - p[nt][3]);
        }
        __syncthreads();
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            if (k0 > r0) break;
            uint32_t a[4];
            frag_a<C>(a, As, r0, k0, lane);
#pragma unroll
            for (int np = 0; np < VSB / 16; np++) {
                uint32_t b[4];
                frag_b_kn<VSB>(b, Us, np * 16, k0, lane);
                mma(ov[np * 2], a, b[0], b[1]);
                mma(ov[np * 2 + 1], a, b[2], b[3]);
            }
        }
        const int t0 = c * C;
#pragma unroll
        for (int nt = 0; nt < VSB / 8; nt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + gq + hf * 8;
                if (t0 + i < m)
                    *reinterpret_cast<float2*>(o + ((size_t)(t0 + i) * s.nv + h) * DK + v0 + nt * 8 + 2 * tq) =
                        make_float2(ov[nt][hf * 2], ov[nt][hf * 2 + 1]);
            }
        if constexpr (PF) {
            __syncthreads();  // all warps are done with Ut / As
            ld_ua(c + 1);
            cp_wait<2>();     // Kh_c
            __syncthreads();
        }
        const float gc = gcs[(size_t)h * nch + c];
#pragma unroll
        for (int nt = 0; nt < KW / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) st[nt][e] *= gc;
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            uint32_t a[4];
            frag_a_t<VSB>(a, Us, sr0, k0, lane);
#pragma unroll
            for (int np = 0; np < KW / 16; np++) {
                uint32_t b[4];
                frag_b_kn<DK>(b, Ks, kb + np * 16, k0, lane);
                mma(st[np * 2], a, b[0], b[1]);
                mma(st[np * 2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
        if constexpr (PF) ld_k(c + 1);
    }
    if constexpr (PF) cp_wait0();
    float* dp = S + (size_t)((cur + m) % kStateSlots) * slot_floats + ((size_t)h * DK + v0) * DK;
#pragma unroll
    for (int nt = 0; nt < KW / 8; nt++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++)
            *reinterpret_cast<float2*>(dp + (size_t)(sr0 + gq + hf * 8) * DK + kb + nt * 8 + 2 * tq) =
                make_float2(st[nt][hf * 2], st[nt][hf * 2 + 1]);
}

thread_local __half* tl_scr = nullptr;
thread_local float* tl_gcs = nullptr;
thread_local size_t tl_scr_n = 0;

}  // namespace

void dn_chunk_reserve(int nv, int m_max) {
    const size_t pairs = (size_t)nv * ((m_max + C - 1) / C);
    qe_ws_reserve("dn_scr", pairs * kPer * 2);
    qe_ws_reserve("dn_gcs", pairs * 4);
}

int dn_chunk_min() {
    static const int v = env_int("QE_DN_CHUNK", 64);
    return v;
}

void dn_chunk_init() {
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_prep_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kPrepSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_prep_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kPrepSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_prep_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kPrepSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kScanSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_scan2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kScanSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_scanv_kernel<32, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)scan_smem<32>()));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_scanv_kernel<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)scan_smem<32>()));
    CUDA_CHECK(cudaFuncSetAttribute(dn_chunk_scanv_kernel<16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)scan_smem<16>()));
}

// qk / gb as written by dn_qk_kernel; writes o and the state after the m tokens (slot cur + m)
void dn_core_chunked(const float* conv_out, int ch, const DeltaShape& s, const float* qk,
                     const float* gb, float* S, size_t slot_floats, const int* cur, int m,
                     float* o, cudaStream_t st, int slot_off, bool h16, const DnConvIn* ci) {
    const int nch = (m + C - 1) / C;
    const size_t need = (size_t)s.nv * nch * kPer;
    if (need > tl_scr_n) {  // grows during the first prefills (never inside graph capture)
        // Flash-Next reserves both at load for its largest chunk (dn_chunk_reserve), so serving takes the reserved
        // blocks; a growth beyond them goes through the engine's out-of-memory hook (drop cached graphs, retry). The
        // old buffers are kept, not freed: graphs captured with them may still be replayed
        size_t nb = qe_ws_alloc((void**)&tl_scr, need * 2, "dn_scr");
        const size_t ng = qe_ws_alloc((void**)&tl_gcs, (size_t)s.nv * nch * 4, "dn_gcs");
        tl_scr_n = std::min(nb / 2 / kPer, ng / 4) * kPer;  // (head, chunk) pairs both buffers hold
    }
    const DnConvIn c0{};
    if (ci)
        dn_chunk_prep_kernel<false, true><<<dim3(s.nv, nch), kPrepThreads, kPrepSmem, st>>>(conv_out, ch, s, qk, gb, m,
                                                                                      tl_scr, tl_gcs, *ci);
    else if (h16)
        dn_chunk_prep_kernel<true><<<dim3(s.nv, nch), kPrepThreads, kPrepSmem, st>>>(conv_out, ch, s, qk, gb, m,
                                                                               tl_scr, tl_gcs, c0);
    else
        dn_chunk_prep_kernel<false><<<dim3(s.nv, nch), kPrepThreads, kPrepSmem, st>>>(conv_out, ch, s, qk, gb, m,
                                                                                tl_scr, tl_gcs, c0);
    // QE_DN_SCAN2: 32 (default) / 16 = v slice per block; 1 = 64-column scan with rolling prefetch; 0 = master's scan
    const int scan2 = env_int("QE_DN_SCAN2", 33);
    if (scan2 == 32 || scan2 == 16 || scan2 == 33) {  // v columns per scan block (32 / 16; 33 = 32 + prefetch)
        if (scan2 == 32)
            dn_chunk_scanv_kernel<32, false><<<dim3(s.nv, DK / 32), kThreads, scan_smem<32>(), st>>>(
                s, tl_scr, tl_gcs, nch, m, S, slot_floats, cur, slot_off, o);
        else if (scan2 == 33)
            dn_chunk_scanv_kernel<32, true><<<dim3(s.nv, DK / 32), kThreads, scan_smem<32>(), st>>>(
                s, tl_scr, tl_gcs, nch, m, S, slot_floats, cur, slot_off, o);
        else
            dn_chunk_scanv_kernel<16, false><<<dim3(s.nv, DK / 16), kThreads, scan_smem<16>(), st>>>(
                s, tl_scr, tl_gcs, nch, m, S, slot_floats, cur, slot_off, o);
    } else if (scan2 == 1)
        dn_chunk_scan2_kernel<<<dim3(s.nv, DK / VS), kThreads, kScanSmem, st>>>(
            s, tl_scr, tl_gcs, nch, m, S, slot_floats, cur, slot_off, o);
    else
        dn_chunk_scan_kernel<<<dim3(s.nv, DK / VS), kThreads, kScanSmem, st>>>(
            s, tl_scr, tl_gcs, nch, m, S, slot_floats, cur, slot_off, o);
}
