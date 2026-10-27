#pragma once
// Prefill W4A8 kernels of Flash-Next (prefill-r3): INT8 activations (per token and G-group of k, scale with 22
// significant bits), weights INT4 unpacked to exact signed int8, mma.m16n8k32 s8 x s8 -> s32 with the accumulator
// started at the bit pattern of M = 1.5 * 2^23 (the int -> float conversion is then an FFMA: fma(f, sx, -M sx) =
// (f - M) * sx rounded once), as the 27B's gemm_i8 kernels.
//
// XQ layout (activations): row-major [rows][H] int8; inside every 64-k chunk lane t (of the MMA quad) owns the 16
// bytes [16t, 16t + 16): for the four 16-k blocks j = 0..3 the values at k = 16j + {2t, 2t + 8, 2t + 1, 2t + 9}
// (the order in which the W4 fragment words hold a row's k: two 16-k words make one k32 A fragment). A k32 step s of
// the chunk uses bytes [8s, 8s + 8) of the lane's 16 (b0 = block 2s, b1 = block 2s + 1).
// Scales: {sx, -M sx} per (row, G-group) (sx = the scale of the true activation, the XH factor 16 included).
// Rows are stored in place of the XH rows: row r's XQ bytes at xq + r * XQ_LD, its scales at xqs + r * XQS_LD.
#include "flash_kernels.cuh"

namespace flash {

constexpr int XQ_LD = 2 * H, XQS_LD = 2 * H / 8;  // bytes / float2 per row
constexpr float kQMagic = 12582912.f;  // 1.5 * 2^23
constexpr uint32_t kQMagicBits = 0x4B400000u;

__device__ __forceinline__ void q_mma(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void q_mma_init(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(kQMagicBits));
}
// 4 nibbles (u = w + 8 at bits 0, 8, 16, 24) -> 4 signed int8 bytes w
__device__ __forceinline__ uint32_t q_nib4(uint32_t q) {
    const uint32_t u = (q & 0x0F0F0F0Fu) | 0x80808080u;
    return (u - 0x08080808u) ^ 0x80808080u;
}
// two 16-k W4 words -> the A fragment of one k32 step (rows g: even nibbles, rows g + 8: odd nibbles)
__device__ __forceinline__ void q_dequant(uint32_t q0, uint32_t q1, uint32_t a[4]) {
    a[0] = q_nib4(q0);
    a[1] = q_nib4(q0 >> 4);
    a[2] = q_nib4(q1);
    a[3] = q_nib4(q1 >> 4);
}
__device__ __forceinline__ float q_scale(float amax) {  // 22 significant bits: M * sx exact in fp32
    return amax > 0.f ? __uint_as_float(__float_as_uint(amax / 127.f) & ~3u) : 1.f;
}

// XH rows (fp16 x / 16, MMA-permuted per 64-k chunk) -> XQ rows + scales. Block = 160 threads (H / 16), one row.
template <int G>
__global__ void __launch_bounds__(160) k_xh_to_q(const __half* __restrict__ xh, int8_t* __restrict__ xq,
                                                 float2* __restrict__ xqs) {
    constexpr int L = G / 16;  // threads per group
    const int r = blockIdx.x, i = threadIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(xh + (size_t)r * H + 16 * i);
    const uint4 p0 = src[0], p1 = src[1];
    const uint32_t w[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    float v[16];
    float mx = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[e]));
        v[2 * e] = f.x;
        v[2 * e + 1] = f.y;
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
#pragma unroll
    for (int o = 1; o < L; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float sc = q_scale(mx);
    const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
    // XH holds block j's values in the order {2t, 2t+1, 2t+8, 2t+9}; XQ wants {2t, 2t+8, 2t+1, 2t+9}
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int q0 = __float2int_rn(v[4 * j] * inv), q1 = __float2int_rn(v[4 * j + 2] * inv);
        const int q2 = __float2int_rn(v[4 * j + 1] * inv), q3 = __float2int_rn(v[4 * j + 3] * inv);
        o[j] = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) | ((uint32_t)(q3 & 0xFF) << 24);
    }
    __syncthreads();  // in place: every thread has read its part of the row
    *reinterpret_cast<uint4*>(xq + (size_t)r * XQ_LD + 16 * i) = make_uint4(o[0], o[1], o[2], o[3]);
    if (i % L == 0) {
        const float sx = sc * kXhScale;  // exact (power of two)
        xqs[(size_t)r * XQS_LD + i / L] = make_float2(sx, -kQMagic * sx);
    }
}

// MoE prefill gate_up, W4A8: one block (8 warps) per job, one pass over K for all the job's rows (<= 64 tokens, NTT
// token tiles of 8: the job body is instantiated per tile count, a predicated-off MMA still costs the tensor pipe);
// warp w owns the 16-row tiles 2w, 2w + 1 (8 gate + 8 up rows each, GLU-interleaved). The tokens' XQ rows are staged
// per 128-k step in shared memory by cp.async (NST stages; 16-byte pieces XOR-swizzled by row parity: conflict-free
// fragment loads), the weights and their scales go straight to registers two steps ahead. act[aoff + t][128] (XH
// layout) = silu(g) * u * w, as k_moe_gu_pf2. Jobs run largest index first (the shared expert's 64-row jobs are last).
constexpr int GQ_TT = 64, GQ_KS = 128, GQ_NST = 4;
template <int G, int NTT, int WT>  // WT: 16-row tiles per warp (2: the whole group in one block, 1: half, rh)
__device__ __forceinline__ void gu_q_job(const PfJob& pj, int rh, const int* __restrict__ tl, const float* __restrict__ wl,
                                         const int8_t* __restrict__ xq, const float2* __restrict__ xqs,
                                         __half* __restrict__ act, int8_t (*xs)[GQ_TT][GQ_KS],
                                         float2 (*ss)[GQ_KS / G][GQ_TT]) {
    constexpr int KS = GQ_KS, NKS = H / KS, NST = GQ_NST, SUB = KS / G, chunks = H >> 6, groups = H >> 7;
    const int m = pj.n;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    // staging: thread -> token row st_r (0..63), 16-byte pieces 2 st_h, 2 st_h + 1 of the 128-byte step slice
    const int st_r = threadIdx.x >> 2, st_h = threadIdx.x & 3;
    const int8_t* xrow = xq + (size_t)tl[pj.off + (st_r < m ? st_r : m - 1)] * XQ_LD;
    const float2* srow = xqs + (size_t)tl[pj.off + ((int)threadIdx.x < m ? (int)threadIdx.x : m - 1)] * XQS_LD;
    auto stage = [&](int ks) {
        const int b = ks % NST;
        if (st_r < 8 * NTT) {
#pragma unroll
            for (int p2 = 0; p2 < 2; p2++) {
                const int p = 2 * st_h + p2, ps = p ^ ((st_r & 1) << 2);
                const unsigned dst = (unsigned)__cvta_generic_to_shared(&xs[b][st_r][ps * 16]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(xrow + ks * KS + p * 16));
            }
        }
        if ((int)threadIdx.x < 8 * NTT) {
#pragma unroll
            for (int s = 0; s < SUB; s++) {
                const unsigned dst = (unsigned)__cvta_generic_to_shared(&ss[b][s][threadIdx.x]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(srow + ks * SUB + s));
            }
        }
        asm volatile("cp.async.commit_group;");
    };
    const int tw0 = WT == 2 ? 2 * warp : 8 * rh + warp;  // first 16-row tile of the warp
    const uint4* qp[WT];
    const __half* sp[WT];
#pragma unroll
    for (int a = 0; a < WT; a++) {
        qp[a] = pj.w.gq + (size_t)(tw0 + a) * chunks * 32 + lane;
        sp[a] = pj.w.gs + (size_t)(tw0 + a) * groups * 16;
    }
    float acc[WT][NTT][4];
#pragma unroll
    for (int a = 0; a < WT; a++)
#pragma unroll
        for (int j = 0; j < NTT; j++) acc[a][j][0] = acc[a][j][1] = acc[a][j][2] = acc[a][j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < NST - 1; s++) stage(s);
    uint4 wn[WT][2], wn2[WT][2];  // weights of the next two steps
    __half sn[WT][2], sn2[WT][2];  // weight scales (rows g, g + 8) of the next two steps
#pragma unroll
    for (int a = 0; a < WT; a++) {
        wn[a][0] = __ldg(qp[a]); wn[a][1] = __ldg(qp[a] + 32);
        wn2[a][0] = __ldg(qp[a] + 64); wn2[a][1] = __ldg(qp[a] + 96);
        sn[a][0] = sp[a][g]; sn[a][1] = sp[a][g + 8];
        sn2[a][0] = sp[a][16 + g]; sn2[a][1] = sp[a][24 + g];
    }
    for (int ks = 0; ks < NKS; ks++) {
        const int b = ks % NST;
        uint4 w[WT][2];
        float sw0[WT], sw8[WT];
#pragma unroll
        for (int a = 0; a < WT; a++) {
            w[a][0] = wn[a][0]; w[a][1] = wn[a][1]; wn[a][0] = wn2[a][0]; wn[a][1] = wn2[a][1];
            sw0[a] = __half2float(sn[a][0]); sw8[a] = __half2float(sn[a][1]);
            sn[a][0] = sn2[a][0]; sn[a][1] = sn2[a][1];
            if (ks + 2 < NKS) {
                wn2[a][0] = __ldg(qp[a] + (size_t)(2 * ks + 4) * 32); wn2[a][1] = __ldg(qp[a] + (size_t)(2 * ks + 5) * 32);
                sn2[a][0] = sp[a][(ks + 2) * 16 + g]; sn2[a][1] = sp[a][(ks + 2) * 16 + g + 8];
            }
        }
        asm volatile("cp.async.wait_group %0;" ::"n"(NST - 2));
        __syncthreads();
        if (ks + NST - 1 < NKS) stage(ks + NST - 1);
        else asm volatile("cp.async.commit_group;");  // keep the group count in step
        uint32_t ci[WT][NTT][4];  // integer sums of the current activation group (+ M)
#pragma unroll
        for (int c = 0; c < 2; c++) {  // 64-k chunk c of the step
            uint32_t A[WT][2][4];      // [tile][k32 step][fragment]
#pragma unroll
            for (int a = 0; a < WT; a++) {
                q_dequant(w[a][c].x, w[a][c].y, A[a][0]);
                q_dequant(w[a][c].z, w[a][c].w, A[a][1]);
            }
            const bool first = c == 0 || G == 64;
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++) {
#pragma unroll
                for (int j = 0; j < NTT; j++) {
                    const int row = 8 * j + g, pc = (c * 4 + t4) ^ ((row & 1) << 2);
                    const uint2 bq = *reinterpret_cast<const uint2*>(&xs[b][row][pc * 16 + 8 * s2]);
#pragma unroll
                    for (int a = 0; a < WT; a++) {
                        if (s2 == 0 && first) q_mma_init(ci[a][j], A[a][0], bq.x, bq.y);
                        else q_mma(ci[a][j], A[a][s2], bq.x, bq.y);
                    }
                }
            }
            if (G == 128 && c == 0) continue;
            // flush the group's integer sums: (f - M) * sx per token, then * sw per row
            const int sg = G == 64 ? c : 0;
#pragma unroll
            for (int j = 0; j < NTT; j++) {
                const float2 x0 = ss[b][sg][8 * j + 2 * t4], x1 = ss[b][sg][8 * j + 2 * t4 + 1];
#pragma unroll
                for (int a = 0; a < WT; a++) {
                    const float f0 = __fmaf_rn(__uint_as_float(ci[a][j][0]), x0.x, x0.y);
                    const float f1 = __fmaf_rn(__uint_as_float(ci[a][j][1]), x1.x, x1.y);
                    const float f2 = __fmaf_rn(__uint_as_float(ci[a][j][2]), x0.x, x0.y);
                    const float f3 = __fmaf_rn(__uint_as_float(ci[a][j][3]), x1.x, x1.y);
                    const float s0 = sw0[a], s8 = sw8[a];
                    acc[a][j][0] = __fmaf_rn(f0, s0, acc[a][j][0]);
                    acc[a][j][1] = __fmaf_rn(f1, s0, acc[a][j][1]);
                    acc[a][j][2] = __fmaf_rn(f2, s8, acc[a][j][2]);
                    acc[a][j][3] = __fmaf_rn(f3, s8, acc[a][j][3]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();  // the next job of this block restages the buffers
#pragma unroll
    for (int a = 0; a < WT; a++) {
        const int col = (tw0 + a) * 8 + g;
#pragma unroll
        for (int j = 0; j < NTT; j++) {
            const int n0 = 8 * j + 2 * t4, n1 = n0 + 1;
            if (n0 < m) store_xh(act + (size_t)(pj.aoff + n0) * GRP, col, silu(acc[a][j][0]) * acc[a][j][2] * wl[pj.off + n0]);
            if (n1 < m) store_xh(act + (size_t)(pj.aoff + n1) * GRP, col, silu(acc[a][j][1]) * acc[a][j][3] * wl[pj.off + n1]);
        }
    }
}
template <int G, int WT = 2>  // WT = 1: two blocks per job (row halves), ~half the registers (2 blocks / SM)
__global__ void __launch_bounds__(256, WT == 2 ? 1 : 2) k_moe_gu_q(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                     const int* __restrict__ tl, const float* __restrict__ wl,
                                                     const int8_t* __restrict__ xq, const float2* __restrict__ xqs,
                                                     __half* __restrict__ act) {
    static_assert(G == 64 || G == 128, "G");
    __shared__ __align__(16) int8_t xs[GQ_NST][GQ_TT][GQ_KS];
    __shared__ __align__(16) float2 ss[GQ_NST][GQ_KS / G][GQ_TT];
    const int nj = *njobs;
    for (int jb = (int)blockIdx.x; jb < nj * (3 - WT); jb += gridDim.x) {
        const int job = nj - 1 - jb / (3 - WT), rh = jb % (3 - WT);
        const PfJob pj = jobs[job];
        switch ((pj.n + 7) >> 3) {
            case 1: gu_q_job<G, 1, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 2: gu_q_job<G, 2, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 3: gu_q_job<G, 3, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 4: gu_q_job<G, 4, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 5: gu_q_job<G, 5, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 6: gu_q_job<G, 6, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            case 7: gu_q_job<G, 7, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
            default: gu_q_job<G, 8, WT>(pj, rh, tl, wl, xq, xqs, act, xs, ss); break;
        }
    }
}

// MoE prefill down, W4A8 (prefill-r3). k_act_to_q quantizes the gate_up output rows in place (fp16 XH 256 bytes per
// row -> XQ 128 bytes + {sx, -M sx} at byte 128: one scale per (token, group) = the down GEMM's 128-k group); then
// k_moe_dn_q as k_moe_dn_slot (grid (H / 128, jobs), 4 warps x 2 16-row output tiles, the first job of an expert runs
// all its local groups) with s8 MMA and one pass over the job's tokens (<= 64, body per 8-token tile count).
__global__ void __launch_bounds__(256) k_act_to_q(__half* __restrict__ act, int rows) {
    const int idx = blockIdx.x * 256 + threadIdx.x, r = idx >> 3, i = idx & 7;  // 8 threads per row
    if (r >= rows) return;  // (rows of 8 threads: whole octets exit together)
    unsigned char* row = reinterpret_cast<unsigned char*>(act + (size_t)r * GRP);
    const uint4* src = reinterpret_cast<const uint4*>(row + 32 * i);
    const uint4 p0 = src[0], p1 = src[1];
    const uint32_t w[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    float v[16];
    float mx = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[e]));
        v[2 * e] = f.x; v[2 * e + 1] = f.y;
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float sc = q_scale(mx), inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
    uint32_t o4[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {  // XH {2t, 2t+1, 2t+8, 2t+9} -> XQ {2t, 2t+8, 2t+1, 2t+9}
        const int q0 = __float2int_rn(v[4 * j] * inv), q1 = __float2int_rn(v[4 * j + 2] * inv);
        const int q2 = __float2int_rn(v[4 * j + 1] * inv), q3 = __float2int_rn(v[4 * j + 3] * inv);
        o4[j] = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) | ((uint32_t)(q3 & 0xFF) << 24);
    }
    __syncwarp();  // in place: the row's 8 threads (one warp) have read it
    *reinterpret_cast<uint4*>(row + 16 * i) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    if (i == 0) { const float sxv = sc * kXhScale; *reinterpret_cast<float2*>(row + GRP) = make_float2(sxv, -kQMagic * sxv); }
}
template <int NTT, int TT>  // TT: 16-row output tiles per warp (block: 4 x TT tiles)
__device__ __forceinline__ void dn_q_job(const PfJob* __restrict__ jobs, int job, int ngj, const __half* __restrict__ act,
                                         SlotBuf slots) {
    const int pn = jobs[job].n, poff = jobs[job].off;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int rb = (blockIdx.x * 4 + warp) * TT;
    const unsigned char* actb = reinterpret_cast<const unsigned char*>(act);
    float acc[TT][NTT][4];
#pragma unroll
    for (int a = 0; a < TT; a++)
#pragma unroll
        for (int j = 0; j < NTT; j++) acc[a][j][0] = acc[a][j][1] = acc[a][j][2] = acc[a][j][3] = 0.f;
    for (int k = 0; k < ngj; k++) {
        const PfJob pj = jobs[job + k];
        uint4 w[TT][2];
        float s0[TT], s8[TT];
#pragma unroll
        for (int a = 0; a < TT; a++) {
            const uint4* qp = pj.w.dq + (size_t)(rb + a) * 2 * 32 + lane;
            w[a][0] = __ldg(qp); w[a][1] = __ldg(qp + 32);
            s0[a] = __half2float(pj.w.ds[(rb + a) * 16 + g]);
            s8[a] = __half2float(pj.w.ds[(rb + a) * 16 + g + 8]);
        }
        uint4 bq[NTT][2];
        float2 x0[NTT], x1[NTT];
#pragma unroll
        for (int j = 0; j < NTT; j++) {
            const int li = 8 * j + g < pn ? 8 * j + g : 0;
            const unsigned char* rr = actb + (size_t)(pj.aoff + li) * GRP * 2;
            bq[j][0] = *reinterpret_cast<const uint4*>(rr + t4 * 16);
            bq[j][1] = *reinterpret_cast<const uint4*>(rr + 64 + t4 * 16);
            const int n0 = 8 * j + 2 * t4;
            x0[j] = *reinterpret_cast<const float2*>(actb + (size_t)(pj.aoff + (n0 < pn ? n0 : 0)) * GRP * 2 + GRP);
            x1[j] = *reinterpret_cast<const float2*>(actb + (size_t)(pj.aoff + (n0 + 1 < pn ? n0 + 1 : 0)) * GRP * 2 + GRP);
        }
        uint32_t ci[TT][NTT][4];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            uint32_t A[TT][2][4];
#pragma unroll
            for (int a = 0; a < TT; a++) {
                q_dequant(w[a][c].x, w[a][c].y, A[a][0]);
                q_dequant(w[a][c].z, w[a][c].w, A[a][1]);
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++)
#pragma unroll
                for (int j = 0; j < NTT; j++) {
                    const uint32_t b0 = s2 ? bq[j][c].z : bq[j][c].x, b1 = s2 ? bq[j][c].w : bq[j][c].y;
#pragma unroll
                    for (int a = 0; a < TT; a++) {
                        if (c == 0 && s2 == 0) q_mma_init(ci[a][j], A[a][0], b0, b1);
                        else q_mma(ci[a][j], A[a][s2], b0, b1);
                    }
                }
        }
#pragma unroll
        for (int j = 0; j < NTT; j++)
#pragma unroll
            for (int a = 0; a < TT; a++) {
                acc[a][j][0] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][0]), x0[j].x, x0[j].y), s0[a], acc[a][j][0]);
                acc[a][j][1] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][1]), x1[j].x, x1[j].y), s0[a], acc[a][j][1]);
                acc[a][j][2] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][2]), x0[j].x, x0[j].y), s8[a], acc[a][j][2]);
                acc[a][j][3] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][3]), x1[j].x, x1[j].y), s8[a], acc[a][j][3]);
            }
    }
#pragma unroll
    for (int j = 0; j < NTT; j++) {
        const int n0 = 8 * j + 2 * t4;
#pragma unroll
        for (int a = 0; a < TT; a++) {
            const int row0 = (rb + a) * 16;
            if (n0 < pn) {
                __half* o = slots.row(poff + n0) + row0 + g;
                o[0] = __float2half_rn(acc[a][j][0] * (1.f / kSlotScale));
                o[8] = __float2half_rn(acc[a][j][2] * (1.f / kSlotScale));
            }
            if (n0 + 1 < pn) {
                __half* o = slots.row(poff + n0 + 1) + row0 + g;
                o[0] = __float2half_rn(acc[a][j][1] * (1.f / kSlotScale));
                o[8] = __float2half_rn(acc[a][j][3] * (1.f / kSlotScale));
            }
        }
    }
}
template <int TT = 2>  // grid (H / 16 / 4 / TT, jobs)
__global__ void __launch_bounds__(128) k_moe_dn_q(const PfJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                  const __half* __restrict__ act, SlotBuf slots) {
    const int job = blockIdx.y;
    if (job >= *njobs) return;
    const int ngj = jobs[job].ng;
    if (ngj == 0) return;
    switch ((jobs[job].n + 7) >> 3) {
        case 1: dn_q_job<1, TT>(jobs, job, ngj, act, slots); break;
        case 2: dn_q_job<2, TT>(jobs, job, ngj, act, slots); break;
        case 3: dn_q_job<3, TT>(jobs, job, ngj, act, slots); break;
        case 4: dn_q_job<4, TT>(jobs, job, ngj, act, slots); break;
        case 5: dn_q_job<5, TT>(jobs, job, ngj, act, slots); break;
        case 6: dn_q_job<6, TT>(jobs, job, ngj, act, slots); break;
        case 7: dn_q_job<7, TT>(jobs, job, ngj, act, slots); break;
        default: dn_q_job<8, TT>(jobs, job, ngj, act, slots); break;
    }
}

// ------------------------------------------------------------------ W8A8 row GEMMs (prefill-r3)
// y[m][N] (fp32) = x[m][K] . W8^T for the INT8 g128 row matrices (QSA qkv / o, router, MTP fc): the weights are
// already int8 (the W8 fragment layout feeds the s8 MMA after two byte permutes per fragment), the activations are
// quantized per (token, 128-k group) by k_x_to_r8 into the R8 layout: per 64-k chunk lane t owns 16 bytes, for the
// two k32 blocks b the values at k = 32b + {2t, 2t+1, 2t+8, 2t+9, 16+2t, 17+2t, 24+2t, 25+2t} (the W8 word order).
// Scales {sx, -M sx} per (token, 128-k group).
// one row per block of exactly K / 16 threads (K % 128 == 0, K <= RQ_KMAX; launch with K / 16): the row is staged
// in shared memory (coalesced 16-byte loads, two per thread: no partial warps in the shuffles), the 128-k group maxima
// by 16-lane shuffles, then thread i writes lane (i & 3) of 64-k chunk i >> 2
constexpr int RQ_KMAX = 4096;
__global__ void __launch_bounds__(RQ_KMAX / 16) k_x_to_r8(const bf16* __restrict__ x, int ldx, int K, int8_t* __restrict__ xq,
                                                        float2* __restrict__ xs) {
    __shared__ __align__(16) bf16 row[RQ_KMAX];
    __shared__ float sc[RQ_KMAX / 128];
    const int r = blockIdx.x, i = threadIdx.x, np = K >> 3;  // 8-value pieces
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * ldx);
    for (int p = i; p < np; p += blockDim.x) {
        const uint4 u = src[p];
        *reinterpret_cast<uint4*>(&row[8 * p]) = u;
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        float mx = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[e]));
            mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if ((p & 15) == 0) sc[p >> 4] = mx;
    }
    __syncthreads();
    const int c = i >> 2, t = i & 3;
    if (c >= (K >> 6)) return;
    const float mx = sc[c >> 1];
    const float s = q_scale(mx);
    const float inv = mx > 0.f ? __frcp_rn(s) : 0.f;
    const bf16* v = row + 64 * c;
    uint32_t o[4];
#pragma unroll
    for (int bb = 0; bb < 2; bb++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int k0 = 32 * bb + 16 * h + 2 * t;  // {k0, k0+1, k0+8, k0+9}
            const float2 f01 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v[k0]));
            const float2 f89 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v[k0 + 8]));
            const int q0 = __float2int_rn(f01.x * inv), q1 = __float2int_rn(f01.y * inv);
            const int q2 = __float2int_rn(f89.x * inv), q3 = __float2int_rn(f89.y * inv);
            o[2 * bb + h] = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) | ((uint32_t)(q3 & 0xFF) << 24);
        }
    *reinterpret_cast<uint4*>(xq + (size_t)r * K + 64 * c + 16 * t) = make_uint4(o[0], o[1], o[2], o[3]);
    if (t == 0 && (c & 1) == 0) xs[(size_t)r * (K >> 7) + (c >> 1)] = make_float2(s, -kQMagic * s);
}
// Block: 8 warps, 128 weight rows (8 tiles of 16) x 128 tokens; warp (wn, wt) = 4 tiles x 32 tokens. Per 128-k stage
// the weights (8 tiles x 2 KB, contiguous in the W8 layout) and the tokens' R8 slices (128 x 128 B, 16-byte pieces
// swizzled by row parity) are staged by cp.async (RQ_NST stages). grid (token blocks, row blocks): the token blocks of
// one weight block run next to each other (L2 reuse). Rows past N (N % 128 != 0) read tile N / 16 - 1, not stored.
constexpr int RQ_TB = 128, RQ_NB = 128, RQ_NST = 3;
constexpr size_t RQ_STAGE = (size_t)RQ_NB * 128 + (size_t)RQ_TB * 128 + (size_t)RQ_TB * 8;  // weights, x, scales
constexpr size_t RQ_SMEM = RQ_NST * RQ_STAGE;
__global__ void __launch_bounds__(256, 1) k_rows_q(const uint4* __restrict__ wq, const bf16* __restrict__ ws, int K, int N,
                                                   const int8_t* __restrict__ xq, const float2* __restrict__ xsc,
                                                   float* __restrict__ y, int ldy, int m) {
    extern __shared__ __align__(16) unsigned char rq_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int wn = warp & 1, wt = warp >> 1;
    const int t0 = blockIdx.x * RQ_TB, nb0 = blockIdx.y * (RQ_NB / 16);  // first token, first tile
    const int ntile = N >> 4, nks = K >> 7, gl = K >> 5;                 // tiles, stages, k32 blocks per tile
    auto sw_ = [&](int b) { return rq_smem + b * RQ_STAGE; };
    auto sx_ = [&](int b) { return rq_smem + b * RQ_STAGE + RQ_NB * 128; };
    auto ss_ = [&](int b) { return reinterpret_cast<float2*>(rq_smem + b * RQ_STAGE + RQ_NB * 128 + RQ_TB * 128); };
    auto stage = [&](int ks) {
        const int b = ks % RQ_NST;
        // weights: 8 tiles x 128 pieces of 16 B (tile-major, as in global); 1024 pieces, 4 per thread
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int pc = threadIdx.x + 256 * i, tl_ = pc >> 7, w = pc & 127;
            const int T = min(nb0 + tl_, ntile - 1);
            const uint4* src = wq + ((size_t)T * gl + ks * 4) * 32 + w;
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sw_(b) + pc * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
        // x: 128 tokens x 8 pieces; 1024 pieces, 4 per thread; piece p of row r at p ^ ((r & 1) << 2)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int pc = threadIdx.x + 256 * i, r = pc >> 3, p = pc & 7;
            const int t = min(t0 + r, m - 1);
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sx_(b) + r * 128 + ((p ^ ((r & 1) << 2)) * 16));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(xq + (size_t)t * K + ks * 128 + p * 16));
        }
        if (threadIdx.x < RQ_TB) {
            const int t = min(t0 + (int)threadIdx.x, m - 1);
            const unsigned dst = (unsigned)__cvta_generic_to_shared(ss_(b) + threadIdx.x);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(xsc + (size_t)t * nks + ks));
        }
        asm volatile("cp.async.commit_group;");
    };
    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[a][j][0] = acc[a][j][1] = acc[a][j][2] = acc[a][j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < RQ_NST - 1; s++)
        if (s < nks) stage(s);
        else asm volatile("cp.async.commit_group;");
    for (int ks = 0; ks < nks; ks++) {
        const int b = ks % RQ_NST;
        asm volatile("cp.async.wait_group %0;" ::"n"(RQ_NST - 2));
        __syncthreads();
        if (ks + RQ_NST - 1 < nks) stage(ks + RQ_NST - 1);
        else asm volatile("cp.async.commit_group;");
        // weight scales of this stage (rows g, g + 8 of the warp's 4 tiles)
        float sw0[4], sw8[4];
#pragma unroll
        for (int a = 0; a < 4; a++) {
            const int T = min(nb0 + 4 * wn + a, ntile - 1);
            sw0[a] = __bfloat162float(ws[((size_t)T * nks + ks) * 16 + g]);
            sw8[a] = __bfloat162float(ws[((size_t)T * nks + ks) * 16 + g + 8]);
        }
        uint32_t ci[4][4][4];
        const unsigned char* wsm = sw_(b);
        const unsigned char* xsm = sx_(b);
#pragma unroll
        for (int cp = 0; cp < 2; cp++) {  // 64-k chunk pairs: one 16-byte x load = 2 k32 blocks
            uint32_t bx[4][4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r = 32 * wt + 8 * j + g, p = (cp * 4 + t4) ^ ((r & 1) << 2);
                const uint4 u = *reinterpret_cast<const uint4*>(xsm + r * 128 + p * 16);
                bx[j][0] = u.x; bx[j][1] = u.y; bx[j][2] = u.z; bx[j][3] = u.w;
            }
#pragma unroll
            for (int kb = 0; kb < 2; kb++) {  // k32 block 2 cp + kb of the stage
                uint32_t A[4][4];
#pragma unroll
                for (int a = 0; a < 4; a++) {
                    const uint4 u = *reinterpret_cast<const uint4*>(wsm + ((size_t)(4 * wn + a) * 128 + (2 * cp + kb) * 32 + lane) * 16);
                    A[a][0] = __byte_perm(u.x, u.y, 0x5410);
                    A[a][1] = __byte_perm(u.x, u.y, 0x7632);
                    A[a][2] = __byte_perm(u.z, u.w, 0x5410);
                    A[a][3] = __byte_perm(u.z, u.w, 0x7632);
                }
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int a = 0; a < 4; a++) {
                        if (cp == 0 && kb == 0) q_mma_init(ci[a][j], A[a], bx[j][0], bx[j][1]);
                        else q_mma(ci[a][j], A[a], bx[j][2 * kb], bx[j][2 * kb + 1]);
                    }
            }
        }
        const float2* ssm = ss_(b);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 x0 = ssm[32 * wt + 8 * j + 2 * t4], x1 = ssm[32 * wt + 8 * j + 2 * t4 + 1];
#pragma unroll
            for (int a = 0; a < 4; a++) {
                acc[a][j][0] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][0]), x0.x, x0.y), sw0[a], acc[a][j][0]);
                acc[a][j][1] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][1]), x1.x, x1.y), sw0[a], acc[a][j][1]);
                acc[a][j][2] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][2]), x0.x, x0.y), sw8[a], acc[a][j][2]);
                acc[a][j][3] = __fmaf_rn(__fmaf_rn(__uint_as_float(ci[a][j][3]), x1.x, x1.y), sw8[a], acc[a][j][3]);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
#pragma unroll
    for (int a = 0; a < 4; a++) {
        const int T = nb0 + 4 * wn + a;
        if (T >= ntile) continue;
        const int row0 = T * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n0 = t0 + 32 * wt + 8 * j + 2 * t4, n1 = n0 + 1;
            if (n0 < m) { y[(size_t)n0 * ldy + row0 + g] = acc[a][j][0]; y[(size_t)n0 * ldy + row0 + g + 8] = acc[a][j][2]; }
            if (n1 < m) { y[(size_t)n1 * ldy + row0 + g] = acc[a][j][1]; y[(size_t)n1 * ldy + row0 + g + 8] = acc[a][j][3]; }
        }
    }
}

// ------------------------------------------------------------------ QSA prefill attention with block selection
// (prefill-r3). k_qsa_dec runs one (row, kv head) per block: every row loads its own ~700 keys (fp8 K / V) and the
// block holds 12 heads in one 16-row MMA tile. Here a block takes PR consecutive rows of one kv head: the union of
// their selected blocks (sorted, ~2.9x one row's at 26K for 16 rows) plus the tail range of their incomplete blocks is
// walked once in 16-key tiles staged in shared memory (prefetched one tile ahead), and the PR x 12 q rows sit in
// 16-row MMA tiles (one per warp, full tiles); a per-key row mask keeps each row to its own keys. Same arithmetic per
// (row, key) as k_qsa_dec (q / 16 in fp16, fp8 keys / values widened exactly, scales x 256, fp32 accumulation); the
// online softmax runs over the keys in another order (union order), so sums round differently.
// grid (NKV, ceil(m / PR), S key splits); smem: qpf_smem(lbmax).
constexpr int QP_R = 8, QP_W = QP_R * (NQ / NKV) / 16, QP_KT = 16;  // rows per block, warps (6), keys per tile
constexpr int QP_UMAX = QP_R * IDX_TOPB;
__host__ __device__ constexpr size_t qpf_kv_bytes() { return (size_t)2 * 2 * QP_KT * QD_SROW * 2; }  // 2 buffers x (K, V)
constexpr int QP_NST = 4;  // fp8 staging ring (cp.async) depth
__host__ __device__ constexpr size_t qpf_raw_bytes() { return (size_t)QP_NST * (2 * QP_KT * HD + 3 * QP_KT * 4); }
__host__ __device__ inline size_t qpf_smem(int lbmax) {
    return qpf_kv_bytes() + qpf_raw_bytes() + (size_t)2 * 3 * QP_KT * 4 + (size_t)QP_UMAX * 4 + QP_UMAX + (size_t)((lbmax + 31) / 32) * 4;
}
// The union of the selected blocks of QP_R consecutive rows (ascending list + each row's bit per entry), once per row
// group (prefill-r4): k_qsa_pfsel built it in every block, i.e. 2 kv heads x S key splits times, ~30 % of its time.
// grid ceil(m / QP_R), 256 threads, smem (lbmax + 31) / 32 words; out: U[g], ublk[g][QP_UMAX], um[g][QP_UMAX]
__global__ void __launch_bounds__(256) k_qpf_union(const int* __restrict__ sel, const int* __restrict__ nsel, int m, int lbmax,
                                                   QpUnion qu) {
    extern __shared__ unsigned qu_bm[];
    __shared__ int s_U, s_ns[QP_R];
    const int t0 = blockIdx.x * QP_R, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nw = (lbmax + 31) / 32;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) qu_bm[i] = 0;
    if (threadIdx.x < QP_R) s_ns[threadIdx.x] = t0 + (int)threadIdx.x < m ? nsel[t0 + threadIdx.x] : 0;
    __syncthreads();
    for (int r = 0; r < QP_R; r++) {
        const int ns = s_ns[r];
        for (int j = threadIdx.x; j < ns; j += blockDim.x) {
            const int b = sel[(size_t)(t0 + r) * lbmax + j];
            atomicOr(&qu_bm[b >> 5], 1u << (b & 31));
        }
    }
    __syncthreads();
    int* const ublk = qu.ublk + (size_t)blockIdx.x * QP_UMAX;
    unsigned char* const um = qu.um + (size_t)blockIdx.x * QP_UMAX;
    if (warp == 0) {
        int run = 0;
        for (int w0 = 0; w0 < nw; w0 += 32) {
            const int w = w0 + lane;
            const int c = w < nw ? __popc(qu_bm[w]) : 0;
            int x = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
            int base = run + x - c;
            if (w < nw) {
                unsigned bits = qu_bm[w];
                while (bits) {
                    const int bb = __ffs(bits) - 1;
                    bits &= bits - 1;
                    ublk[base++] = w * 32 + bb;
                }
            }
            run += __shfl_sync(0xffffffffu, x, 31);
        }
        if (lane == 0) s_U = run;
    }
    __syncthreads();
    const int U = s_U;
    // each row's bits: rank of a selected block = popcount of the bitmap words before it (prefix over words)
    __syncthreads();
    for (int u = threadIdx.x; u < U; u += blockDim.x) um[u] = 0;
    __syncthreads();
    for (int r = 0; r < QP_R; r++) {
        const int ns = s_ns[r];
        for (int j = threadIdx.x; j < ns; j += blockDim.x) {
            const int b = sel[(size_t)(t0 + r) * lbmax + j];
            int lo = 0, hi = U - 1;
            while (lo < hi) { const int mid = (lo + hi) >> 1; if (ublk[mid] < b) lo = mid + 1; else hi = mid; }
            atomicOr(reinterpret_cast<unsigned*>(um + (lo & ~3)), (1u << r) << (8 * (lo & 3)));
        }
    }
    if (threadIdx.x == 0) qu.U[blockIdx.x] = U;
}
template <bool PVH, bool QKH>  // fp16-accumulate P.V (per 16-key tile, flushed into fp32) / Q.K^T
__global__ void __launch_bounds__(QP_W * 32, 1) k_qsa_pfsel(const float* __restrict__ qb, const uint8_t* __restrict__ kc,
                                                            const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                            const float* __restrict__ vs, const int* __restrict__ pos0,
                                                            AttnDst dst, Plan p, Xchg xq, const int* __restrict__ sel,
                                                            const int* __restrict__ nsel, int lbmax, int m,
                                                            const int* __restrict__ rbase, bool q16, int dbg = 0,
                                                            QpUnion qu = QpUnion(), int ngrp = 0) {
    extern __shared__ __align__(16) unsigned char qp_sm[];
    constexpr int HPG = NQ / NKV, NTH = QP_W * 32;
    __half* KV = reinterpret_cast<__half*>(qp_sm);  // [buf][K | V][QP_KT][QD_SROW]
    unsigned char* raw = qp_sm + qpf_kv_bytes();  // [QP_NST][K | V][QP_KT][HD] fp8, then [QP_NST][kscale | vscale | mask][QP_KT]
    float* rmeta = reinterpret_cast<float*>(raw + (size_t)QP_NST * 2 * QP_KT * HD);
    float* kmeta = reinterpret_cast<float*>(qp_sm + qpf_kv_bytes() + qpf_raw_bytes());  // [buf][kscale | vscale | mask bits][QP_KT]
    int* ublk = reinterpret_cast<int*>(kmeta + 2 * 3 * QP_KT);
    unsigned char* um = reinterpret_cast<unsigned char*>(ublk + QP_UMAX);
    unsigned* bm = reinterpret_cast<unsigned*>(um + QP_UMAX);
    __shared__ unsigned s_s;
    __shared__ int s_rc[QP_R], s_nr[QP_R], s_ns[QP_R], s_U, s_wp[33];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int nw = (lbmax + 31) / 32;
    if (rbase) { const size_t rb = rbase[0]; kc += rb * NKV * HD; vc += rb * NKV * HD; ks += rb * NKV; vs += rb * NKV; }
    // persistent (prefill-r4, qu.ctr set): a grid of one block per SM takes the (kv head, key split, row group) items
    // from a counter (wave quantization: 256 items on 82 SMs were 4 waves, the last one 10 % full); else one item per
    // block (grid NKV x groups x S)
    const bool pers = qu.ctr != nullptr;
    const int S = pers ? qu.S : (int)gridDim.z;
    const unsigned total = (unsigned)NKV * (pers ? ngrp : (int)gridDim.y) * S;
    __shared__ unsigned s_item;
    for (bool first = true;; first = false) {
    int kvh, t0, z;
    if (pers) {
        __syncthreads();
        if (threadIdx.x == 0) s_item = atomicAdd(qu.ctr, 1u);
        __syncthreads();
        const unsigned it = s_item;
        if (it >= total) break;
        kvh = it % NKV, z = (it / NKV) % S, t0 = (int)(it / (NKV * S)) * QP_R;
    } else {
        if (!first) break;
        kvh = blockIdx.x, t0 = blockIdx.y * QP_R, z = blockIdx.z;
    }
    const int gy = t0 / QP_R;
    __syncthreads();
    for (int i = threadIdx.x; i < nw; i += NTH) bm[i] = 0;
    if (threadIdx.x < QP_R) {
        const int r = threadIdx.x, t = t0 + r;
        if (t < m) {
            const int pos = *pos0 + t;
            const int ns = nsel[t];
            const int nra = owned_upto(p, pos);
            s_nr[r] = nra;
            s_rc[r] = ns < 0 ? nra : owned_upto(p, IDX_R * ((pos + 1) / IDX_R) - 1);
            s_ns[r] = ns;  // -1: no selection (every visible key: rows < nra as "tail")
            if (ns < 0) s_rc[r] = 0;
        } else {
            s_nr[r] = 0; s_rc[r] = 0; s_ns[r] = 0;
        }
    }
    if (threadIdx.x == 0 && dbg == 3) s_s = *xq.seq;  // shadow run after k_qsa_dec consumed the exchange (debug)
    else if (threadIdx.x == 0) {  // the peers' q rows of this kv head's heads
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
    if (dbg == 4) {  // debug: handshake only
        if (threadIdx.x == 0) {
            __threadfence();
            if (atomicAdd(xq.done, 1u) == total - 1) { *xq.done = 0; *xq.seq = s_s; __threadfence(); }
        }
        continue;
    }
    // union of the rows' selected blocks (bitmap -> ascending list), then each row's bit per union entry: precomputed
    // per row group by k_qpf_union (qu.U set), else built here
    if (qu.U) {
        if (threadIdx.x == 0) s_U = __ldcg(qu.U + gy);
        __syncthreads();
        const int U0 = s_U;
        const int* gb = qu.ublk + (size_t)gy * QP_UMAX;
        const unsigned char* gm = qu.um + (size_t)gy * QP_UMAX;
        for (int u = threadIdx.x; u < U0; u += NTH) ublk[u] = __ldcg(gb + u);
        for (int u = threadIdx.x; u < (U0 + 3) / 4; u += NTH)
            reinterpret_cast<unsigned*>(um)[u] = __ldcg(reinterpret_cast<const unsigned*>(gm) + u);
        __syncthreads();
    } else {
    // union of the rows' selected blocks (bitmap -> ascending list), then each row's bit per union entry
        for (int r = 0; r < QP_R; r++) {
            const int ns = s_ns[r];
            for (int j = threadIdx.x; j < ns; j += NTH) {
                const int b = sel[(size_t)(t0 + r) * lbmax + j];
                atomicOr(&bm[b >> 5], 1u << (b & 31));
            }
        }
        __syncthreads();
        if (warp == 0) {  // word prefix counts (32 words per lane step)
            int run = 0;
            for (int w0 = 0; w0 < nw; w0 += 32) {
                const int w = w0 + lane;
                const int c = w < nw ? __popc(bm[w]) : 0;
                int x = c;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
                // exclusive prefix of word w: run + x - c; store into the list by expanding the word's bits
                int base = run + x - c;
                if (w < nw) {
                    unsigned bits = bm[w];
                    while (bits) {
                        const int bb = __ffs(bits) - 1;
                        bits &= bits - 1;
                        ublk[base++] = w * 32 + bb;
                    }
                }
                run += __shfl_sync(0xffffffffu, x, 31);
            }
            if (lane == 0) s_U = run;
        }
        __syncthreads();
        const int Ub = s_U;
        for (int u = threadIdx.x; u < Ub; u += NTH) um[u] = 0;
        __syncthreads();
        for (int r = 0; r < QP_R; r++) {  // rank of each selected block in the union list (binary search)
            const int ns = s_ns[r];
            for (int j = threadIdx.x; j < ns; j += NTH) {
                const int b = sel[(size_t)(t0 + r) * lbmax + j];
                int lo = 0, hi = Ub - 1;
                while (lo < hi) { const int mid = (lo + hi) >> 1; if (ublk[mid] < b) lo = mid + 1; else hi = mid; }
                um[lo] |= (unsigned char)(1u << r);
            }
            __syncthreads();
        }
    }
    const int U = s_U;
    int tmin = 1 << 30, tmax = 0;
#pragma unroll
    for (int r = 0; r < QP_R; r++)
        if (s_nr[r] > s_rc[r]) { tmin = min(tmin, s_rc[r]); tmax = max(tmax, s_nr[r]); }
    const int ntail = tmax > tmin ? tmax - tmin : 0;
    const int NK = 4 * U + ntail;
    const int ntile = (NK + QP_KT - 1) / QP_KT;
    const int tile0 = z * ntile / S, tile1 = dbg == 1 ? tile0 : (z + 1) * ntile / S;
    if (dbg == 2 && threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned*>(dst.o[7]), (unsigned)ntile);
    // Q fragments: q rows qi = 16 warp + g / + 8 -> row qi / 12, head kvh * 12 + qi % 12
    uint32_t qa[HD / 16][4];
    {
        auto qptr = [&](int qi, const float*& f, const __half*& hp) {
            const int r = qi / HPG, h = kvh * HPG + qi % HPG, t = t0 + r;
            f = nullptr; hp = nullptr;
            if (t >= m) return;
            const int owner = h / QPC(p);
            if (owner == p.rank) f = qb + ((size_t)t * (NQ + 4) + h) * HD;
            else if (q16) hp = reinterpret_cast<const __half*>(xq.recv[p.rank] + ((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS * qper(p)) +
                               (size_t)t * qper(p) + (h - owner * QPC(p)) * HD;
            else f = xq.recv[p.rank] + (((size_t)(s_s & (XSLOT - 1)) * p.ngpu + owner) * XROWS + t) * qper(p) + (h - owner * QPC(p)) * HD;
        };
        const float *f0, *f1;
        const __half *h0, *h1;
        qptr(16 * warp + g, f0, h0);
        qptr(16 * warp + g + 8, f1, h1);
        auto ld2 = [&](const float* f, const __half* hp, int c) -> uint32_t {
            if (hp) return __ldcg(reinterpret_cast<const unsigned*>(hp + c));
            if (!f) return 0u;
            const float2 a = __ldcg(reinterpret_cast<const float2*>(f + c));
            return f2h2(a.x * (1.f / 16.f), a.y * (1.f / 16.f));
        };
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const int c = kk * 16 + 2 * t4;
            qa[kk][0] = ld2(f0, h0, c);
            qa[kk][1] = ld2(f1, h1, c);
            qa[kk][2] = ld2(f0, h0, c + 8);
            qa[kk][3] = ld2(f1, h1, c + 8);
        }
    }
    const int r0 = (16 * warp + g) / HPG, r1 = (16 * warp + g + 8) / HPG;
    // tile loader: 16 keys x (K, V) x 16 pieces of 16 bytes = 512 pieces; piece q: key q >> 5, K / V (q >> 4) & 1, c q & 15
    constexpr int NPC = (2 * QP_KT * 16 + NTH - 1) / NTH;
    auto key_of = [&](int kidx, int& kr, unsigned& msk) {
        if (kidx >= NK) { kr = 0; msk = 0; return; }
        if (kidx < 4 * U) { kr = ublk[kidx >> 2] * IDX_R + (kidx & 3); msk = um[kidx >> 2]; return; }
        kr = tmin + (kidx - 4 * U);
        msk = 0;
#pragma unroll
        for (int r = 0; r < QP_R; r++) msk |= (kr >= s_rc[r] && kr < s_nr[r]) ? (1u << r) : 0u;
    };
    // cp.async of tile `tile` (fp8 K / V rows, their scales; keys no row uses: zero-filled) into ring slot tile % QP_NST
    auto issue = [&](int tile) {
        if (tile < tile1) {
            const int sl = tile % QP_NST;
            unsigned char* rb = raw + (size_t)sl * 2 * QP_KT * HD;
#pragma unroll
            for (int i = 0; i < NPC; i++) {
                const int q = threadIdx.x + i * NTH;
                if (q < 2 * QP_KT * 16) {
                    const int key = q >> 5, kv = (q >> 4) & 1, c = q & 15;
                    int kr; unsigned msk;
                    key_of(tile * QP_KT + key, kr, msk);
                    const uint8_t* src = (kv ? vc : kc) + ((size_t)(msk ? kr : 0) * NKV + kvh) * HD + c * 16;
                    const unsigned dst = (unsigned)__cvta_generic_to_shared(rb + ((size_t)kv * QP_KT + key) * HD + c * 16);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(msk ? 16 : 0));
                }
            }
            if (threadIdx.x < 2 * QP_KT) {
                const int key = threadIdx.x & (QP_KT - 1), kv = threadIdx.x / QP_KT;
                int kr; unsigned msk;
                key_of(tile * QP_KT + key, kr, msk);
                float* rm = rmeta + (size_t)sl * 3 * QP_KT;
                const float* src = (kv ? vs : ks) + (size_t)(msk ? kr : 0) * NKV + kvh;
                const unsigned dst = (unsigned)__cvta_generic_to_shared(rm + kv * QP_KT + key);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(msk ? 4 : 0));
                if (kv == 0) rm[2 * QP_KT + key] = __uint_as_float(msk);
            }
        }
        asm volatile("cp.async.commit_group;");
    };
    // ring slot of `tile` -> fp16 buffer buf (fp8 widened exactly; scales x 256)
    auto convert = [&](int tile, int buf) {
        const int sl = tile % QP_NST;
        const unsigned char* rb = raw + (size_t)sl * 2 * QP_KT * HD;
        __half* Kb = KV + (size_t)buf * 2 * QP_KT * QD_SROW;
#pragma unroll
        for (int i = 0; i < NPC; i++) {
            const int q = threadIdx.x + i * NTH;
            if (q < 2 * QP_KT * 16) {
                const int key = q >> 5, kv = (q >> 4) & 1, c = q & 15;
                const uint4 w = *reinterpret_cast<const uint4*>(rb + ((size_t)kv * QP_KT + key) * HD + c * 16);
                uint32_t h[8];
                f8x4_h(w.x, h[0], h[1]); f8x4_h(w.y, h[2], h[3]); f8x4_h(w.z, h[4], h[5]); f8x4_h(w.w, h[6], h[7]);
                uint4* d = reinterpret_cast<uint4*>(Kb + ((size_t)kv * QP_KT + key) * QD_SROW + c * 16);
                d[0] = make_uint4(h[0], h[1], h[2], h[3]);
                d[1] = make_uint4(h[4], h[5], h[6], h[7]);
            }
        }
        if (threadIdx.x < QP_KT) {
            const int key = threadIdx.x;
            const float* rm = rmeta + (size_t)sl * 3 * QP_KT;
            float* km = kmeta + buf * 3 * QP_KT;
            km[key] = rm[key] * 256.f;
            km[QP_KT + key] = rm[QP_KT + key] * 256.f;
            km[2 * QP_KT + key] = rm[2 * QP_KT + key];
        }
    };
    float o[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < QP_NST - 1; s2++) issue(tile0 + s2);
    asm volatile("cp.async.wait_group %0;" ::"n"(QP_NST - 2));
    __syncthreads();
    if (tile0 < tile1) convert(tile0, 0);
    __syncthreads();
    for (int tile = tile0; tile < tile1; tile++) {
        const int buf = (tile - tile0) & 1;
        issue(tile + QP_NST - 1);
        const __half* Ks = KV + (size_t)buf * 2 * QP_KT * QD_SROW;
        const __half* Vs = Ks + QP_KT * QD_SROW;
        const float* km = kmeta + buf * 3 * QP_KT;
        float sc[2][4];
        if constexpr (QKH) {  // fp16 accumulators, two chains per n-tile (even / odd k-steps)
            uint32_t hs[2][2][2];
#pragma unroll
            for (int j = 0; j < 2; j++) hs[j][0][0] = hs[j][0][1] = hs[j][1][0] = hs[j][1][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk++) {
                uint32_t b[4];
                ta_ldsm4(b, Ks + ((lane & 7) + ((lane >> 4) << 3)) * QD_SROW + kk * 16 + ((lane >> 3) & 1) * 8);
                xh_mma(hs[0][kk & 1], qa[kk], b[0], b[1]);
                xh_mma(hs[1][kk & 1], qa[kk], b[2], b[3]);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const float2 a0 = __half22float2(*reinterpret_cast<__half2*>(&hs[j][0][0])), a1 = __half22float2(*reinterpret_cast<__half2*>(&hs[j][1][0]));
                const float2 b0 = __half22float2(*reinterpret_cast<__half2*>(&hs[j][0][1])), b1 = __half22float2(*reinterpret_cast<__half2*>(&hs[j][1][1]));
                sc[j][0] = a0.x + a1.x; sc[j][1] = a0.y + a1.y; sc[j][2] = b0.x + b1.x; sc[j][3] = b0.y + b1.y;
            }
        } else {
#pragma unroll
        for (int j = 0; j < 2; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            uint32_t b[4];
            ta_ldsm4(b, Ks + ((lane & 7) + ((lane >> 4) << 3)) * QD_SROW + kk * 16 + ((lane >> 3) & 1) * 8);
            mma16(sc[0], qa[kk], b[0], b[1]);
            mma16(sc[1], qa[kk], b[2], b[3]);
        }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = j * 8 + t4 * 2 + e;
                const unsigned msk = __float_as_uint(km[2 * QP_KT + key]);
                const float kf = km[key];
                sc[j][e] = (msk >> r0) & 1 ? sc[j][e] * kf : -INFINITY;
                sc[j][2 + e] = (msk >> r1) & 1 ? sc[j][2 + e] * kf : -INFINITY;
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
        if constexpr (!PVH) {
#pragma unroll
            for (int j = 0; j < HD / 8; j++) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
        }
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const float va = km[QP_KT + j * 8 + t4 * 2], vb = km[QP_KT + j * 8 + t4 * 2 + 1];
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
            if constexpr (PVH) {  // the tile's 16 keys in fp16, then o = o * c + tile (one FFMA per element)
                uint32_t h0[2] = {0u, 0u}, h1[2] = {0u, 0u};
                xh_mma(h0, a, b[0], b[1]);
                xh_mma(h1, a, b[2], b[3]);
                const float2 p0 = __half22float2(*reinterpret_cast<__half2*>(&h0[0])), q0 = __half22float2(*reinterpret_cast<__half2*>(&h0[1]));
                const float2 p1 = __half22float2(*reinterpret_cast<__half2*>(&h1[0])), q1 = __half22float2(*reinterpret_cast<__half2*>(&h1[1]));
                o[2 * dp][0] = __fmaf_rn(o[2 * dp][0], c0, p0.x); o[2 * dp][1] = __fmaf_rn(o[2 * dp][1], c0, p0.y);
                o[2 * dp][2] = __fmaf_rn(o[2 * dp][2], c1, q0.x); o[2 * dp][3] = __fmaf_rn(o[2 * dp][3], c1, q0.y);
                o[2 * dp + 1][0] = __fmaf_rn(o[2 * dp + 1][0], c0, p1.x); o[2 * dp + 1][1] = __fmaf_rn(o[2 * dp + 1][1], c0, p1.y);
                o[2 * dp + 1][2] = __fmaf_rn(o[2 * dp + 1][2], c1, q1.x); o[2 * dp + 1][3] = __fmaf_rn(o[2 * dp + 1][3], c1, q1.y);
            } else {
                mma16(o[2 * dp], a, b[0], b[1]);
                mma16(o[2 * dp + 1], a, b[2], b[3]);
            }
        }
        asm volatile("cp.async.wait_group %0;" ::"n"(QP_NST - 2));
        __syncthreads();
        if (tile + 1 < tile1) convert(tile + 1, buf ^ 1);
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    float* out = dst.o[z];
    const int qi0 = 16 * warp + g, qi1 = qi0 + 8;
    const int ta = t0 + qi0 / HPG, tb = t0 + qi1 / HPG;
    const int ha = kvh * HPG + qi0 % HPG, hb = kvh * HPG + qi1 % HPG;
#pragma unroll
    for (int j = 0; j < HD / 8; j++) {
        const int c = j * 8 + t4 * 2;
        if (ta < m) *reinterpret_cast<float2*>(out + ((size_t)ta * NQ + ha) * (HD + 2) + c) = make_float2(o[j][0], o[j][1]);
        if (tb < m) *reinterpret_cast<float2*>(out + ((size_t)tb * NQ + hb) * (HD + 2) + c) = make_float2(o[j][2], o[j][3]);
    }
    if (t4 == 0) {
        if (ta < m) { out[((size_t)ta * NQ + ha) * (HD + 2) + HD] = m0; out[((size_t)ta * NQ + ha) * (HD + 2) + HD + 1] = l0; }
        if (tb < m) { out[((size_t)tb * NQ + hb) * (HD + 2) + HD] = m1; out[((size_t)tb * NQ + hb) * (HD + 2) + HD + 1] = l1; }
    }
    __syncthreads();
    if (threadIdx.x == 0 && dbg != 3) {
        __threadfence();
        if (atomicAdd(xq.done, 1u) == total - 1) {
            *xq.done = 0;
            *xq.seq = s_s;
            __threadfence();
        }
    }
    }  // items
    if (pers && threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(qu.ctr + 1, 1u) == gridDim.x - 1) { qu.ctr[1] = 0; qu.ctr[0] = 0; __threadfence(); }
    }
}
// debug (FLASH_QPF_CMP): normalized partials o / L of two attention outputs; st: {max rel err bits, row, head}
__global__ void k_cmp_attn(const float* __restrict__ a, const float* __restrict__ b, int m, unsigned* __restrict__ st) {
    const int t = blockIdx.x;
    for (int h = 0; h < NQ; h++) {
        const float* x = a + ((size_t)t * NQ + h) * (HD + 2);
        const float* y = b + ((size_t)t * NQ + h) * (HD + 2);
        const float la = x[HD + 1], lb = y[HD + 1];
        float e = 0.f, mx = 1e-6f;
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            const float u = la > 0.f ? x[d] / la : 0.f, v = lb > 0.f ? y[d] / lb : 0.f;
            e = fmaxf(e, fabsf(u - v));
            mx = fmaxf(mx, fabsf(u));
        }
        for (int o = 16; o > 0; o >>= 1) { e = fmaxf(e, __shfl_xor_sync(0xffffffffu, e, o)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
        if ((threadIdx.x & 31) == 0) {
            const float r = (la > 0.f) != (lb > 0.f) ? 1e9f : e / mx;
            const unsigned bits = __float_as_uint(isnan(r) ? 1e30f : r);
            atomicMax(st, bits);
            if (r > 2e-2f) { const unsigned k = atomicAdd(st + 1, 1u); if (k < 16) st[4 + k] = (unsigned)t * 32u + h; }
        }
    }
}

// debug (FLASH_SELSTAT): per 16 consecutive rows, the union of their selected local blocks vs the sum of the rows'
// counts -> st[0] += sum of counts, st[1] += union, st[2] += groups (rows with a selection only)
__global__ void k_sel_union(const int* __restrict__ sel, const int* __restrict__ nsel, int m, int lbmax,
                            unsigned long long* __restrict__ st) {
    __shared__ unsigned bm[65536 / 32];
    __shared__ unsigned cnt;
    const int r0 = blockIdx.x * QP_R;
    const int nb = min(lbmax, 65536);
    for (int i = threadIdx.x; i < (nb + 31) / 32; i += blockDim.x) bm[i] = 0;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    unsigned long long tot = 0;
    bool any = false;
    for (int r = r0; r < min(m, r0 + QP_R); r++) {
        const int n = nsel[r];
        if (n < 0) continue;
        any = true;
        tot += n;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const int b = sel[(size_t)r * lbmax + j];
            if (b < nb) atomicOr(&bm[b >> 5], 1u << (b & 31));
        }
    }
    __syncthreads();
    unsigned c = 0;
    for (int i = threadIdx.x; i < (nb + 31) / 32; i += blockDim.x) c += __popc(bm[i]);
    atomicAdd(&cnt, c);
    __syncthreads();
    if (threadIdx.x == 0 && any) { atomicAdd(st, tot); atomicAdd(st + 1, (unsigned long long)cnt); atomicAdd(st + 2, 1ull); }
}

}  // namespace flash
