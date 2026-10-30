// Tensor-core attention (head_dim 256) for prefill chunks and for decode / MTP verification.
//
// One kernel template covers both:
//   prefill: block = 64 query rows of one local q head (4 warps x 16 rows), keys [0, causal end)
//   decode : block = one sequence split of one kv head; rows = (head in GQA group, new token)
//            pairs padded to NW*16, writes unnormalized partials for attn_combine
// Keys stream from the kv cache in tiles of 32. With an FP8 (e4m3) cache, tiles arrive as fp8
// through a double-buffered cp.async stage (half the bytes) and are widened to bf16 in shared
// memory; the per-tensor k/v scales are folded into the score scale and the output.
#include "../env.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <functional>
#include <vector>
#include <cfloat>

#include "ops.cuh"
#include "watchdog.cuh"

namespace {

#ifndef QE_PBKV
#define QE_PBKV 32
#endif
#ifndef QE_PABL  // ablation bits (bench only): 1 no widen, 2 no tile loads, 4 no exp, 8 no QK mma, 16 no PV mma
#define QE_PABL 0
#endif
constexpr int D = 256;
constexpr int ROWCH = D / 8;   // 16-byte chunks per bf16 row
constexpr int ROWCH8 = D / 16; // 16-byte chunks per fp8 row

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem),
                 "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

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
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_hacc(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool H>
__device__ __forceinline__ void mma_t(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    if (H) mma_f16(c, a, b0, b1);
    else mma(c, a, b0, b1);
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    bf162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
template <bool H>
__device__ __forceinline__ uint32_t pack_t(float a, float b) {
    return H ? pack_f16(a, b) : pack_bf16(a, b);
}

// 512-byte bf16 rows: swizzle the low 3 bits of the 16-byte chunk index with the row
__device__ __forceinline__ int sw(int row, int ch) { return row * ROWCH + (ch ^ (row & 7)); }

// Two e4m3 bytes (in the low byte of each 16-bit lane of t) -> two fp16 equal to x * 2^-8:
// fp16's exponent bias is 8 larger, so placing the bits is enough, subnormals included.
// The 2^8 is folded into the k / v scales.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t t) {
    return ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
}
__device__ __forceinline__ uint4 widen8(uint32_t w0, uint32_t w1) {  // 8 fp8 -> 8 fp16
    return make_uint4(e4m3x2_to_f16x2(__byte_perm(w0, 0, 0x4140)),
                      e4m3x2_to_f16x2(__byte_perm(w0, 0, 0x4342)),
                      e4m3x2_to_f16x2(__byte_perm(w1, 0, 0x4140)),
                      e4m3x2_to_f16x2(__byte_perm(w1, 0, 0x4342)));
}

// Prefill keeps Q in registers (QR): its shared-memory staging area is reused for the K/V tiles,
// so with 16-key tiles a block needs 32 KB and two blocks share an SM.
// BKV keys per tile: 16 for prefill and fp8 decode (two blocks per SM), else 32.
// fp8 prefill: 32-key tiles with a single fp8 staging buffer (the next tile's copy starts once
// the current one is widened), 48 KB per block -> still two blocks per SM.
template <bool F8, bool DEC>
constexpr int bkv_for() { return (F8 && !DEC) ? QE_PBKV : (F8 || !DEC) ? 16 : 32; }
template <bool F8, int BKV>
constexpr bool single_stage() { return F8 && BKV >= 32; }

template <int NW, bool F8, int BKV>
constexpr size_t kv_bytes() {  // 16-bit K/V (two stages; fp8: one widened tile + two fp8 stages)
    return F8 ? 2 * BKV * D * 2 + (single_stage<F8, BKV>() ? 1 : 2) * 2 * BKV * D : 2 * 2 * BKV * D * 2;
}
template <int NW, bool F8, int BKV, bool QR = false>
constexpr size_t smem_bytes() {
    const size_t q = (size_t)NW * 16 * D * 2;
    return QR ? (q > kv_bytes<NW, F8, BKV>() ? q : kv_bytes<NW, F8, BKV>())
              : q + kv_bytes<NW, F8, BKV>();
}

// DECODE = false: prefill. NW compute warps of 16 rows each; blocks have at least 4 warps so
// tile loads and fp8 widening are spread over enough threads. With fp8 the math runs in fp16
// (tiles widen to fp16 cheaply; Q is converted once).
template <int NW>
constexpr int threads_for() { return (NW < 4 ? 4 : NW) * 32; }

template <int NW, bool F8, bool DECODE, int BKV = bkv_for<F8, DECODE>(), bool QR = !DECODE>
__global__ void __launch_bounds__(threads_for<NW>(), QR ? 2 : 1) attn_tc_kernel(
    const bf16* __restrict__ q, const void* __restrict__ kc_, const void* __restrict__ vc_,
    AttnShape s, const float* __restrict__ proj, int ld, const int* __restrict__ pos, int m,
    bf16* __restrict__ out, float* __restrict__ work) {
    extern __shared__ __align__(128) uint4 sm[];
    constexpr int ROWS = NW * 16, NT = threads_for<NW>();
    constexpr bool H = F8;  // fp16 math
    uint4* qs = sm;                    // [ROWS][D] bf16 (QR: only until Q is in registers)
    uint4* kb16 = QR ? sm : qs + ROWS * ROWCH;   // 16-bit K tile(s)
    uint4* vb16 = kb16 + (F8 ? 1 : 2) * BKV * ROWCH;
    uint4* k8 = vb16 + (F8 ? 1 : 2) * BKV * ROWCH;  // fp8 staging [2][BKV][D]
    constexpr bool ONE = single_stage<F8, BKV>();
    uint4* v8 = k8 + (ONE ? 1 : 2) * BKV * ROWCH8;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0];

    // ---- rows, key range (local key space of this GPU's share of the kv head: kv-pos-split) ----
    int kv, h = 0, q0 = 0, split = 0, G, kbeg, kend;
    if (DECODE) {
        kv = blockIdx.x;
        split = blockIdx.y;
        G = (s.kv_qe[kv] - s.kv_qb[kv]) * m;
    } else {
        h = blockIdx.y;
        q0 = blockIdx.x * ROWS;
        kv = s.q2kv[h];
        G = min(ROWS, m - q0);
    }
    const PosMap pm = s.kv_pm[kv];
    const int p0l = pm_rank(pm, p0), splitl = pm_rank(pm, s.split);
    if (DECODE) {
        // split boundaries depend only on the position (not on m), so a token's attention is
        // computed identically however many draft tokens are verified with it
        const int L = pm_rank(pm, p0 + m);
        int chunk = (p0l + kMaxGemvTokens + kAttnSplits - 1) / kAttnSplits;
        chunk = (chunk + BKV - 1) / BKV * BKV;
        kbeg = split * chunk;
        kend = min(L, kbeg + chunk);
    } else {
        kbeg = 0;
        kend = pm_rank(pm, p0 + min(q0 + ROWS, m));
        if (gridDim.z > 1) {  // split-K over the key range (few query tiles): partial outputs
            split = blockIdx.z;
            int chunk = (kend + gridDim.z - 1) / gridDim.z;
            chunk = (chunk + BKV - 1) / BKV * BKV;
            kbeg = split * chunk;
            kend = min(kend, kbeg + chunk);
        }
    }
    // partial outputs: split-K, or a head whose keys are split with a peer (exchange + combine)
    const bool part = !DECODE && (gridDim.z > 1 || s.kv_x[kv] != 0);
    // prefill partials: work = [split][m][nq][256] then ml [split][m][nq][2]
    auto pacc = [&](int row) {
        return work + (((size_t)split * m + q0 + row) * s.nq + h) * 256;
    };
    auto pml = [&](int row) {
        return work + (size_t)gridDim.z * m * s.nq * 256 + (((size_t)split * m + q0 + row) * s.nq + h) * 2;
    };
    if (part && kbeg >= kend) {
        for (int r = tid; r < G; r += NT) {
            pml(r)[0] = -FLT_MAX;
            pml(r)[1] = 0.f;
        }
        return;
    }
    float* wacc = nullptr;
    float* wml = nullptr;
    if (DECODE) {
        wacc = work + ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 256;
        wml = work + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256 +
              ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 2;
        if (kbeg >= kend) {
            for (int r = tid; r < G; r += NT) {
                wml[r * 2] = -FLT_MAX;
                wml[r * 2 + 1] = 0.f;
            }
            return;
        }
    }
    // row r -> (q row in qbuf, query position)
    auto row_q = [&](int r, int& qrow, int& qpos) {
        if (DECODE) {
            const int rr = min(r, G - 1);
            const int hl = rr / m, mi = rr % m;
            qrow = mi * s.nq + s.kv_qb[kv] + hl;
            qpos = p0 + mi;
        } else {
            const int qi = q0 + min(r, G - 1);
            qrow = qi * s.nq + h;
            qpos = p0 + qi;
        }
    };
    for (int i = tid; i < ROWS * ROWCH; i += NT) {
        const int r = i / ROWCH, ch = i % ROWCH;
        int qrow, qp;
        row_q(r, qrow, qp);
        cp_async16(&qs[sw(r, ch)], q + (size_t)qrow * D + ch * 8, r < G);
    }

    // ---- kv tile pipeline ----
    const size_t esz = F8 ? 1 : 2;
    // keys [0, split) in the shared prefix region, the rest in the own region at key - split
    // (local rows of this GPU's share of the head)
    const char* kb = (const char*)kc_ + head_rows(s, kv, s.max_t) * D * esz;
    const char* vb = (const char*)vc_ + head_rows(s, kv, s.max_t) * D * esz;
    const char* kb2 = (const char*)s.kc2 + head_rows(s, kv, s.split) * D * esz;
    const char* vb2 = (const char*)s.vc2 + head_rows(s, kv, s.split) * D * esz;
    const int ntiles = (kend - kbeg + BKV - 1) / BKV;
    auto load_tile = [&](int buf, int tile) {
        const int k0 = kbeg + tile * BKV;
        constexpr int CH = F8 ? ROWCH8 : ROWCH;
        for (int i = tid; i < BKV * CH; i += NT) {
            const int r = i / CH, ch = i % CH;
            const int key = k0 + r;
            const bool ok = key < kend;
            const int kk = min(key, kend - 1);
            const bool pre = kk < splitl;
            const size_t off = ((size_t)(pre ? kk : kk - splitl) * D) * esz + ch * 16;
            const char* kp = (pre ? kb2 : kb) + off;
            const char* vp = (pre ? vb2 : vb) + off;
            if (F8) {
                cp_async16(&k8[buf * BKV * ROWCH8 + i], kp, ok);
                cp_async16(&v8[buf * BKV * ROWCH8 + i], vp, ok);
            } else {
                cp_async16(&kb16[buf * BKV * ROWCH + sw(r, ch)], kp, ok);
                cp_async16(&vb16[buf * BKV * ROWCH + sw(r, ch)], vp, ok);
            }
        }
    };
    // QR: Q -> registers once (fp16 for the fp8 path), then the area is free for K/V tiles
    uint32_t qf[QR ? D / 16 : 1][4];
    if (QR) {
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (warp < NW) {
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                ldsm_x4(qf[kk], &qs[sw(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))]);
                if (F8)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float2 f = __bfloat1622float2(*reinterpret_cast<bf162*>(&qf[kk][j]));
                        qf[kk][j] = pack_f16(f.x, f.y);
                    }
            }
        }
        __syncthreads();
    }
    load_tile(0, 0);
    cp_commit();

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[j][e] = 0.f;
    float mx[2] = {-FLT_MAX, -FLT_MAX}, l[2] = {0.f, 0.f};
    const int r0 = warp * 16 + g, r1 = r0 + 8;
    int qpos0, qpos1, tmp;
    row_q(r0, tmp, qpos0);
    row_q(r1, tmp, qpos1);
    const float kscale = F8 ? s.ks * 256.f : 1.f, vscale = F8 ? s.vs * 256.f : 1.f;
    const float scale = 0.0625f * 1.4426950408889634f * kscale;  // 1/sqrt(256) * log2(e) * ks

    for (int tile = 0; tile < ntiles; tile++) {
        if (ONE) {
            cp_wait<0>();
        } else {
            if (tile + 1 < ntiles) load_tile((tile + 1) & 1, tile + 1);
            cp_commit();
            cp_wait<1>();
        }
        __syncthreads();
        if (F8 && !QR && tile == 0) {  // Q bf16 -> fp16 in place, once
            for (int i = tid; i < ROWS * ROWCH; i += NT) {
                uint4 v = qs[i];
                uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<bf162*>(&w[j]));
                    w[j] = pack_f16(f.x, f.y);
                }
                qs[i] = v;
            }
        }
        const uint4* kt;
        const uint4* vt;
        if (F8) {  // widen this tile's fp8 stage into the bf16 tile
            const uint4* ks8 = k8 + (ONE ? 0 : (tile & 1)) * BKV * ROWCH8;
            const uint4* vs8 = v8 + (ONE ? 0 : (tile & 1)) * BKV * ROWCH8;
            for (int i = tid; i < BKV * ROWCH8 && !(!DECODE && (QE_PABL & 1)); i += NT) {
                const int r = i / ROWCH8, c = i % ROWCH8;
                const uint4 a = ks8[i], b = vs8[i];
                kb16[sw(r, 2 * c)] = widen8(a.x, a.y);
                kb16[sw(r, 2 * c + 1)] = widen8(a.z, a.w);
                vb16[sw(r, 2 * c)] = widen8(b.x, b.y);
                vb16[sw(r, 2 * c + 1)] = widen8(b.z, b.w);
            }
            __syncthreads();
            if (ONE) {  // staging is free: copy the next tile while this one is computed
                if (tile + 1 < ntiles && !(!DECODE && (QE_PABL & 2))) load_tile(0, tile + 1);
                cp_commit();
            }
            kt = kb16;
            vt = vb16;
        } else {
            kt = kb16 + (tile & 1) * BKV * ROWCH;
            vt = vb16 + (tile & 1) * BKV * ROWCH;
        }

        if (warp < NW) {
        // S = Q K^T : 16 rows x BKV keys per warp
        constexpr int NJ = BKV / 8;
        float sc[NJ][4];
        // fp16 prefill over an fp8 cache: P V uses fp16-accumulate MMA (twice the fp32-accumulate
        // rate on GA102) per 16-key tile, added to fp32; Q K^T stays fp32-accumulate (raw scores
        // have a large range: fp16 accumulation there measurably changed the logits)
#ifndef QE_HACC
#define QE_HACC 1
#endif
#ifndef QE_HACC_S
#define QE_HACC_S 1
#endif
        constexpr bool HACC = QE_HACC && F8 && !DECODE;
        constexpr bool HACC_S = QE_HACC_S && F8 && !DECODE;
        uint32_t sch[NJ][2];
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            sch[j][0] = sch[j][1] = 0u;
#pragma unroll
            for (int e = 0; e < 4; e++) sc[j][e] = 0.f;
        }
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            uint32_t a[4];
            if (QR) {
                a[0] = qf[kk][0]; a[1] = qf[kk][1]; a[2] = qf[kk][2]; a[3] = qf[kk][3];
            } else {
                ldsm_x4(a, &qs[sw(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))]);
            }
#pragma unroll
            for (int jp = 0; jp < NJ / 2; jp++) {
                const int mi = lane >> 3;
                uint32_t b[4];
                ldsm_x4(b, &kt[sw((jp * 2 + (mi >> 1)) * 8 + (lane & 7), kk * 2 + (mi & 1))]);
                if (HACC_S && (QE_PABL & 8)) {
                    sch[jp * 2][0] ^= b[0] ^ a[0]; sch[jp * 2 + 1][1] ^= b[3] ^ a[1];
                } else if (HACC_S) {
                    mma_hacc(sch[jp * 2], a, b[0], b[1]);
                    mma_hacc(sch[jp * 2 + 1], a, b[2], b[3]);
                } else {
                    mma_t<H>(sc[jp * 2], a, b[0], b[1]);
                    mma_t<H>(sc[jp * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        if (HACC_S)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&sch[j][0]));
                const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&sch[j][1]));
                sc[j][0] = lo.x; sc[j][1] = lo.y; sc[j][2] = hi.x; sc[j][3] = hi.y;
            }
        // causal mask + online softmax (rows g and g+8; a quad of lanes shares a row); only
        // tiles reaching the new tokens (local row >= p0l) need the absolute position
        const int k0 = kbeg + tile * BKV;
        const bool tail = k0 + BKV > p0l;
        float tmax[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int key = k0 + j * 8 + 2 * t + (e & 1);
                const int qp = (e < 2) ? qpos0 : qpos1;
                float v = sc[j][e] * scale;
                if (key >= kend || (tail && pm_unrank(pm, key) > qp)) v = -FLT_MAX;
                sc[j][e] = v;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], v);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
            const float nm = fmaxf(mx[r], tmax[r]);
            alpha[r] = (mx[r] == -FLT_MAX) ? 0.f : exp2f(mx[r] - nm);
            mx[r] = nm;
        }
        float rs[2] = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = sc[j][e];
                const float p = (v == -FLT_MAX) ? 0.f : ((QE_PABL & 4) && !DECODE) ? (v - mx[e >> 1]) : exp2f(v - mx[e >> 1]);
                sc[j][e] = p;
                rs[e >> 1] += p;
            }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
            rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
            l[r] = l[r] * alpha[r] + rs[r];
        }
        if (HACC) {  // o = o * alpha + (P V of the whole tile in fp16)
            uint32_t a[BKV / 16][4];
#pragma unroll
            for (int ks16 = 0; ks16 < BKV / 16; ks16++) {
                a[ks16][0] = pack_f16(sc[ks16 * 2][0], sc[ks16 * 2][1]);
                a[ks16][1] = pack_f16(sc[ks16 * 2][2], sc[ks16 * 2][3]);
                a[ks16][2] = pack_f16(sc[ks16 * 2 + 1][0], sc[ks16 * 2 + 1][1]);
                a[ks16][3] = pack_f16(sc[ks16 * 2 + 1][2], sc[ks16 * 2 + 1][3]);
            }
#pragma unroll
            for (int jp = 0; jp < 16; jp++) {
                const int mi = lane >> 3;
                uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u};
#pragma unroll
                for (int ks16 = 0; ks16 < BKV / 16; ks16++) {
                    uint32_t b[4];
                    ldsm_x4_t(b, &vt[sw(ks16 * 16 + (mi & 1) * 8 + (lane & 7), jp * 2 + (mi >> 1))]);
                    if (QE_PABL & 16) { c0[0] ^= b[0] ^ a[ks16][0]; c1[1] ^= b[3] ^ a[ks16][1]; continue; }
                    mma_hacc(c0, a[ks16], b[0], b[1]);
                    mma_hacc(c1, a[ks16], b[2], b[3]);
                }
                const float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0]));
                const float2 x1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1]));
                const float2 y0 = __half22float2(*reinterpret_cast<__half2*>(&c1[0]));
                const float2 y1 = __half22float2(*reinterpret_cast<__half2*>(&c1[1]));
                float* p0 = o[jp * 2];
                float* p1 = o[jp * 2 + 1];
                p0[0] = p0[0] * alpha[0] + x0.x; p0[1] = p0[1] * alpha[0] + x0.y;
                p0[2] = p0[2] * alpha[1] + x1.x; p0[3] = p0[3] * alpha[1] + x1.y;
                p1[0] = p1[0] * alpha[0] + y0.x; p1[1] = p1[1] * alpha[0] + y0.y;
                p1[2] = p1[2] * alpha[1] + y1.x; p1[3] = p1[3] * alpha[1] + y1.y;
            }
        } else {
#pragma unroll
        for (int j = 0; j < 32; j++) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        // O += P V : P from registers (A layout), V via ldmatrix.trans
#pragma unroll
        for (int ks16 = 0; ks16 < BKV / 16; ks16++) {
            uint32_t a[4];
            a[0] = pack_t<H>(sc[ks16 * 2][0], sc[ks16 * 2][1]);
            a[1] = pack_t<H>(sc[ks16 * 2][2], sc[ks16 * 2][3]);
            a[2] = pack_t<H>(sc[ks16 * 2 + 1][0], sc[ks16 * 2 + 1][1]);
            a[3] = pack_t<H>(sc[ks16 * 2 + 1][2], sc[ks16 * 2 + 1][3]);
#pragma unroll
            for (int jp = 0; jp < 16; jp++) {
                const int mi = lane >> 3;
                uint32_t b[4];
                ldsm_x4_t(b, &vt[sw(ks16 * 16 + (mi & 1) * 8 + (lane & 7), jp * 2 + (mi >> 1))]);
                mma_t<H>(o[jp * 2], a, b[0], b[1]);
                mma_t<H>(o[jp * 2 + 1], a, b[2], b[3]);
            }
        }
        }
        }  // warp < NW
        __syncthreads();
    }
    cp_wait<0>();
    if (warp >= NW) return;

#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = r == 0 ? r0 : r1;
        if (row >= G) continue;
        if (part) {
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int d = j * 8 + 2 * t;
                *reinterpret_cast<float2*>(pacc(row) + d) =
                    make_float2(o[j][r * 2] * vscale, o[j][r * 2 + 1] * vscale);
            }
            if (t == 0) {
                pml(row)[0] = mx[r] == -FLT_MAX ? -FLT_MAX : mx[r] * 0.6931471805599453f;
                pml(row)[1] = l[r];
            }
        } else if (DECODE) {  // unnormalized partials, natural-log max for attn_combine
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int d = j * 8 + 2 * t;
                *reinterpret_cast<float2*>(wacc + (size_t)row * 256 + d) =
                    make_float2(o[j][r * 2] * vscale, o[j][r * 2 + 1] * vscale);
            }
            if (t == 0) {
                wml[row * 2] = mx[r] == -FLT_MAX ? -FLT_MAX : mx[r] * 0.6931471805599453f;
                wml[row * 2 + 1] = l[r];
            }
        } else {  // normalize, gate, store bf16 (an o_proj head; the others are sent as partials)
            const int qi = q0 + row;
            const int no = s.no ? s.no : s.nq;
            if (h < s.ob || h >= s.ob + no) continue;
            const float inv = vscale / l[r];
            const float* gate = proj + (size_t)qi * ld + h * 512 + 256;
            bf16* op = out + ((size_t)qi * no + (h - s.ob)) * D;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int d = j * 8 + 2 * t;
                const float g0 = 1.f / (1.f + __expf(-gate[d]));
                const float g1 = 1.f / (1.f + __expf(-gate[d + 1]));
                *reinterpret_cast<bf162*>(op + d) =
                    __floats2bfloat162_rn(o[j][r * 2] * inv * g0, o[j][r * 2 + 1] * inv * g1);
            }
        }
    }
}

// ---------------------------------------------------------------- fp8 prefill attention, 8 warps
// attn_pf8_kernel (default for fp8 prefill; QE_PATTN_V2=0 -> attn_tc_kernel). Block = 128 query rows of
// one q head (8 warps x 16 rows), one block per SM, 64-key tiles, 96 KB of shared memory:
//   K fp16 [64][256] | V fp16 [64][256] | K fp8 [64][256] | V fp8 [64][256]
// The fp8 -> fp16 widening of a tile is done once for 8 warps and overlapped with the MMAs: while the
// warps compute Q K^T of tile t they widen V(t) (its buffer was released by P V of tile t-1), while they
// compute P V(t) they widen K(t+1) (released by Q K^T(t)). Every thread widens exactly the 16-byte chunks
// it copied itself, so the fp8 stages need only cp.async.wait_group of its own copies; two barriers per
// tile. fp8 -> fp16 in 7 integer ops per 4 bytes (sign | magnitude >> 1 of a byte in the high half of a
// 16-bit lane), which yields dims (0,2),(1,3) of each group of four: Q is staged with the same dim
// permutation (and pre-scaled by the softmax scale) and the output is permuted back. Softmax in fp16
// (the scores are fp16 MMA results already): P = 2^(S - m) as half2 are directly the A fragments of P V,
// and the row sums come from the same fp16 P through an extra MMA against a ones column (numerator and
// denominator see identical P). P V accumulates the whole 64-key tile in fp16, then o = o * alpha + c.
#ifndef QE_PF8_DBG  // 1: fp32 softmax + fp32 row sums (precision comparisons)
#define QE_PF8_DBG 0
#endif
#ifndef QE_PF8_EARLY  // 1: widen a whole tile at the start of its phase and issue the next copy right after
#define QE_PF8_EARLY 0
#endif
#ifndef QE_PF8_ABL  // ablation bits (bench only): 1 no widen, 2 no Q K^T mma, 4 no P V mma, 8 no tile copies
#define QE_PF8_ABL 0
#endif
constexpr int kP8Rows = 128, kP8Keys = 64, kP8Threads = 256;
// uint4 offsets; QK8 (int8 Q K^T): K int8 [64][256] instead of fp16, + per-key score scales [64] float2
template <bool QK8, int BK> constexpr int p8_v16() { return QK8 ? BK * 16 : BK * 32; }
template <bool QK8, int BK> constexpr int p8_k8() { return p8_v16<QK8, BK>() + BK * 32; }
template <bool QK8, int BK> constexpr int p8_v8() { return p8_k8<QK8, BK>() + BK * 16; }
template <bool QK8, int BK> constexpr int p8_sk() { return p8_v8<QK8, BK>() + BK * 16; }
template <bool QK8, int BK> constexpr size_t p8_smem() { return (size_t)(p8_sk<QK8, BK>() + (QK8 ? BK / 2 : 0)) * 16; }
constexpr int kP8K16 = 0;
__device__ __forceinline__ void imma(uint32_t c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// fp16 x -> int8 round(x * inv) in the low byte of each half (x * inv + 1536 rounds to an integer:
// fp16's ulp in [1024, 2048) is 1, and |x * inv| <= 127)
__device__ __forceinline__ uint32_t h2_q8(uint32_t x, uint32_t inv2) {
    const __half2 r = __hfma2(*reinterpret_cast<__half2*>(&x), *reinterpret_cast<__half2*>(&inv2), __float2half2_rn(1536.f));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// four e4m3 bytes b0..b3 -> e = (b0, b2), o = (b1, b3) as fp16 (value * 2^-8, subnormals included)
__device__ __forceinline__ void f8x4_to_f16(uint32_t q, uint32_t& e, uint32_t& o) {
    o = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 1);
    const uint32_t s = q << 8;
    e = (s & 0x80008000u) | ((s & 0x7F007F00u) >> 1);
}
// cp.async with a memory clobber: the widening reads the fp8 stage with plain loads that must not
// sink below the copy of the next tile into the same stage (no barrier in between)
__device__ __forceinline__ void cp_async16m(void* smem, const void* gmem, bool pred) {
    const unsigned sa = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(gmem), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void ldsm_x4a(uint32_t r[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4ta(uint32_t r[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t h2_exp2(uint32_t x) {
    uint32_t y;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) {
    __half2 r = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t h2_max(uint32_t a, uint32_t b) {
    __half2 r = __hmax2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ float h2_maxf(uint32_t a) {
    const float2 f = __half22float2(*reinterpret_cast<__half2*>(&a));
    return fmaxf(f.x, f.y);
}

template <bool QK8, int NW>
__global__ void __launch_bounds__(NW * 32, NW == 8 ? 1 : 2) attn_pf8_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    AttnShape s, const float* __restrict__ proj, int ld, const int* __restrict__ pos, int m,
    bf16* __restrict__ out, float* __restrict__ work) {
    extern __shared__ __align__(128) uint4 sm[];
    // NW = 8: 128 rows, 64-key tiles, one block per SM; NW = 4: 64 rows, 32-key tiles, two blocks per SM
    constexpr int kP8Rows = NW * 16, kP8Keys = NW * 8, kP8Threads = NW * 32, KS = kP8Threads / 16, NJ = kP8Keys / 8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0];
    // 1-D grid, longest blocks first: q heads in order of decreasing key share (kv-pos-split: whole
    // heads before the position-split ones), then split, then query tile
    const int QT = (m + kP8Rows - 1) / kP8Rows, S = s.pf_splits;
    const int qt = blockIdx.x % QT, sp = (blockIdx.x / QT) % S, hh = blockIdx.x / (QT * S);
    int h = 0;
    {
        int kvo[kMaxLocalKv];
        for (int j = 0; j < s.nkv; j++) kvo[j] = j;
        for (int a = 1; a < s.nkv; a++)  // insertion sort by A / P (descending), stable
            for (int b = a; b > 0; b--) {
                const PosMap& x = s.kv_pm[kvo[b]];
                const PosMap& y = s.kv_pm[kvo[b - 1]];
                if ((long long)x.A * y.P > (long long)y.A * x.P) { const int tt = kvo[b]; kvo[b] = kvo[b - 1]; kvo[b - 1] = tt; }
                else break;
            }
        int rem = hh;
        for (int j = 0; j < s.nkv; j++) {
            const int kj = kvo[j], n = s.kv_qe[kj] - s.kv_qb[kj];
            if (rem < n) { h = s.kv_qb[kj] + rem; break; }
            rem -= n;
        }
    }
    const int q0 = qt * kP8Rows, kv = s.q2kv[h];
    const int G = min(kP8Rows, m - q0);
    const PosMap pm = s.kv_pm[kv];
    const int p0l = pm_rank(pm, p0), splitl = pm_rank(pm, s.split);
    int kbeg = 0, kend = pm_rank(pm, p0 + min(q0 + kP8Rows, m));
    const int split = sp;
    if (S > 1) {
        int chunk = (kend + S - 1) / S;
        chunk = (chunk + kP8Keys - 1) / kP8Keys * kP8Keys;
        kbeg = split * chunk;
        kend = min(kend, kbeg + chunk);
    }
    const bool part = S > 1 || s.kv_x[kv] != 0;
    auto pacc = [&](int row) { return work + (((size_t)split * m + q0 + row) * s.nq + h) * 256; };
    auto pml = [&](int row) {
        return work + (size_t)S * m * s.nq * 256 + (((size_t)split * m + q0 + row) * s.nq + h) * 2;
    };
    if (part && kbeg >= kend) {
        for (int r = tid; r < G; r += kP8Threads) {
            pml(r)[0] = -FLT_MAX;
            pml(r)[1] = 0.f;
        }
        return;
    }
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    constexpr int kP8V16 = p8_v16<QK8, kP8Keys>(), kP8K8 = p8_k8<QK8, kP8Keys>(), kP8V8 = p8_v8<QK8, kP8Keys>(), kP8SK = p8_sk<QK8, kP8Keys>();
    __shared__ float s_sq[QK8 ? kP8Rows : 1];  // QK8: per-row Q scale (q_scaled = int8 * sq)
    __shared__ float s_cq[QK8 ? kP8Rows : 1];  // QK8 + K smoothing: q_scaled . mu (exact fp32, added to the scores)
    // QK8 + K smoothing: K is quantized as k - mu (mu = per-channel mean of the call's keys, fp16, from
    // s.kmu sums); score = int8 (q, k - mu) + q . mu, the second term per row in fp32
    const bool smooth = QK8 && s.kmu != nullptr;
    const float* kmu = smooth ? s.kmu + (size_t)kv * D : nullptr;
    const float kmu_inv = smooth ? 1.f / (float)max(1, pm_rank(pm, p0 + m)) : 0.f;

    // ---- Q: permuted dims, fp16, pre-scaled by 1/sqrt(256) * log2(e) * k scale -> registers
    // (QK8: int8 in natural dim order with one scale per block: amax / 127)
    if (QK8) {
        const float sc = 0.0625f * 1.4426950408889634f * s.ks * 256.f;
        for (int i = tid; i < kP8Rows * 16; i += kP8Threads) {  // 16 dims -> one 16-byte chunk; a row = 16 lanes
            const int r = i >> 4, ch = i & 15;
            const int qi = q0 + min(r, G - 1);
            const uint4* src = reinterpret_cast<const uint4*>(q + ((size_t)qi * s.nq + h) * D) + 2 * ch;
            const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float am = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {  // words
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[j]));
                am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o2));
            const float qa = am * fabsf(sc);
            const float sq = qa > 0.f ? qa / 127.f : 1.f;
            const float qinv = sc / sq;
            if (ch == 0) s_sq[r] = sq;
            if (smooth) {
                float cq = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[j]));
                    const int d = ch * 16 + 2 * j;
                    cq += f.x * __half2float(__float2half_rn(kmu[d] * kmu_inv)) +
                          f.y * __half2float(__float2half_rn(kmu[d + 1] * kmu_inv));
                }
#pragma unroll
                for (int o2 = 1; o2 < 16; o2 <<= 1) cq += __shfl_xor_sync(0xffffffffu, cq, o2);
                if (ch == 0) s_cq[r] = cq * sc;
            }
            uint32_t o4[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f0 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j]));
                const float2 f1 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j + 1]));
                const int b0 = __float2int_rn(f0.x * qinv), b1 = __float2int_rn(f0.y * qinv);
                const int b2 = __float2int_rn(f1.x * qinv), b3 = __float2int_rn(f1.y * qinv);
                o4[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((uint32_t)(b3 & 0xFF) << 24);
            }
            sm[r * 16 + (ch ^ (r & 7))] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
    } else {
        const float sc = 0.0625f * 1.4426950408889634f * s.ks * 256.f;
        for (int i = tid; i < kP8Rows * 32; i += kP8Threads) {
            const int r = i >> 5, ch = i & 31;
            const int qi = q0 + min(r, G - 1);
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(q + ((size_t)qi * s.nq + h) * D) + ch);
            const float2 a0 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.x));
            const float2 a1 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.y));
            const float2 a2 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.z));
            const float2 a3 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.w));
            sm[sw(r, ch)] = make_uint4(pack_f16(a0.x * sc, a1.x * sc), pack_f16(a0.y * sc, a1.y * sc),
                                       pack_f16(a2.x * sc, a3.x * sc), pack_f16(a2.y * sc, a3.y * sc));
        }
    }
    __syncthreads();
    constexpr int QKK = QK8 ? D / 32 : D / 16;  // Q K^T k-steps
    uint32_t qf[QKK][4];
#pragma unroll
    for (int kk = 0; kk < QKK; kk++) {
        const int r = warp * 16 + (lane & 15), ch = kk * 2 + (lane >> 4);
        ldsm_x4(qf[kk], QK8 ? &sm[r * 16 + (ch ^ (r & 7))] : &sm[sw(r, ch)]);
    }
    __syncthreads();

    // ---- tile copies / widening: thread owns chunk (key cr + 16 u, 16-byte column cc), u = 0..3
    const int cr = tid >> 4, cc = tid & 15;
    const uint8_t* kb = kc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* vb = vc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* kb2 = (const uint8_t*)s.kc2 + head_rows(s, kv, s.split) * D;
    const uint8_t* vb2 = (const uint8_t*)s.vc2 + head_rows(s, kv, s.split) * D;
    const int ntiles = (kend - kbeg + kP8Keys - 1) / kP8Keys;
    auto load = [&](bool isv, int tile) {
        if ((QE_PF8_ABL & 8) && tile > 0) return;
        const int k0 = kbeg + tile * kP8Keys;
        if ((QE_PF8_ABL & 256) && !isv) {  // gate for precomputed K only: K int8 straight into its compute buffer
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int key = min(k0 + cr + KS * u, kend - 1);
                cp_async16m(sm + (QK8 ? 0 : kP8K16) + tid + kP8Threads * u, kb + (size_t)max(0, key - splitl) * D + cc * 16, true);
            }
            return;
        }
        if (QE_PF8_ABL & 128) {  // gate for precomputed tiles: K 1 B/elem, V 2 B/elem straight into the compute buffers
            const uint8_t* own = isv ? vb : kb;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int key = min(k0 + cr + KS * u, kend - 1);
                const uint8_t* src = own + (size_t)max(0, key - splitl) * D + cc * 16;
                if (!isv) {
                    cp_async16m(sm + (QK8 ? 0 : kP8K16) + tid + kP8Threads * u, src, true);
                } else {
                    cp_async16m(sm + kP8V16 + tid + 2 * kP8Threads * u, src, true);
                    cp_async16m(sm + kP8V16 + tid + 2 * kP8Threads * u + kP8Threads, src, true);
                }
            }
            return;
        }
        uint4* dst = sm + (isv ? kP8V8 : kP8K8) + tid;
        const uint8_t* own = isv ? vb : kb;
        const uint8_t* pre = isv ? vb2 : kb2;
        if (k0 + kP8Keys <= kend && (k0 >= splitl || k0 + kP8Keys <= splitl)) {
            const uint8_t* src = (k0 >= splitl ? own + (size_t)(k0 - splitl) * D : pre + (size_t)k0 * D) +
                                 (size_t)cr * D + cc * 16;
#pragma unroll
            for (int u = 0; u < 4; u++) cp_async16m(dst + kP8Threads * u, src + (size_t)u * KS * D, true);
        } else {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int key = k0 + cr + KS * u;
                const int kk = min(key, kend - 1);
                const uint8_t* src = (kk < splitl ? pre + (size_t)kk * D : own + (size_t)(kk - splitl) * D) + cc * 16;
                cp_async16m(dst + kP8Threads * u, src, key < kend);
            }
        }
    };
    const int wsw0 = cr * 32 + ((2 * cc) ^ (cr & 7)), wsw1 = cr * 32 + ((2 * cc + 1) ^ (cr & 7));
    const int wsw8 = cr * 16 + (cc ^ (cr & 7));
    // QK8: per-key int8 scales of the K tile in the fp8 stage, all 4 units at once (independent shuffle
    // chains): max |k| from the raw e4m3 bytes (magnitude order = integer order of b & 0x7F; two bytes per
    // half2 as fp16 subnormals, which HMNMX2 orders like integers), inv = 127 / max as half2 per unit
    uint32_t kinv[QK8 ? 4 : 1];
    uint32_t mue[QK8 ? 4 : 1], muo[QK8 ? 4 : 1];  // mu of this thread's 16 dims as (d0, d2), (d1, d3) half2 per word
    if (QK8) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (tid & 15) * 16 + 4 * j;
            auto mu = [&](int dd) { return smooth ? __float2half_rn(kmu[dd] * kmu_inv) : __float2half_rn(0.f); };
            const __half2 a2 = __halves2half2(mu(d), mu(d + 2)), b2 = __halves2half2(mu(d + 1), mu(d + 3));
            mue[j] = *reinterpret_cast<const uint32_t*>(&a2);
            muo[j] = *reinterpret_cast<const uint32_t*>(&b2);
        }
    }
    auto kscales = [&](int tile) {
        if (s.ksc) {  // precomputed by pattn_kscale_kernel: (inv as half2, scale) per local key row
            const uint2* kr = s.ksc + (size_t)kv * s.ksc_ld;
            const int k0 = kbeg + tile * kP8Keys;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint2 v = __ldg(kr + min(k0 + cr + KS * u, kend - 1));
                kinv[u] = v.x;
                if (cc == 0) {
                    const float skq = __uint_as_float(v.y);
                    reinterpret_cast<float2*>(sm + kP8SK)[cr + KS * u] = make_float2(skq, -(12582912.f * skq));
                }
            }
            return;
        }
        int km[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 a = sm[kP8K8 + tid + kP8Threads * u];
            const uint32_t w[4] = {a.x, a.y, a.z, a.w};
            uint32_t hm = 0u;
#pragma unroll
            for (int j = 0; j < 4; j++) hm = h2_max(hm, h2_max(w[j] & 0x007F007Fu, (w[j] >> 8) & 0x007F007Fu));
            km[u] = max((int)(hm & 0xFFFF), (int)(hm >> 16));
        }
#pragma unroll
        for (int o2 = 1; o2 < 16; o2 <<= 1)
#pragma unroll
            for (int u = 0; u < 4; u++) km[u] = max(km[u], __shfl_xor_sync(0xffffffffu, km[u], o2));
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const float kmax = __half2float(__ushort_as_half((unsigned short)(km[u] << 7)));  // the byte as fp16 * 2^-8
            const __half ih = __float2half_rn(kmax > 0.f ? fminf(127.f / kmax, 65504.f) : 0.f);
            const __half2 ih2 = __halves2half2(ih, ih);
            kinv[u] = *reinterpret_cast<const uint32_t*>(&ih2);
            if (cc == 0) {
                const float fi = __half2float(ih);
                const float skq = fi > 0.f ? 1.f / fi : 0.f;
                reinterpret_cast<float2*>(sm + kP8SK)[cr + KS * u] = make_float2(skq, -(12582912.f * skq));
            }
        }
    };
    auto widen = [&](bool isv, int u) {
        if (QE_PF8_ABL & 1) return;
        if ((QE_PF8_ABL & 256) && !isv) return;
        if (QK8 && !isv) {  // K -> int8 with the per-key scale of kscales()
            const uint4 a = sm[kP8K8 + tid + kP8Threads * u];
            uint32_t e[4], o[4];
            f8x4_to_f16(a.x, e[0], o[0]);
            f8x4_to_f16(a.y, e[1], o[1]);
            f8x4_to_f16(a.z, e[2], o[2]);
            f8x4_to_f16(a.w, e[3], o[3]);
            uint32_t w[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (smooth) { e[j] = h2_sub(e[j], mue[j]); o[j] = h2_sub(o[j], muo[j]); }
                w[j] = __byte_perm(h2_q8(e[j], kinv[u]), h2_q8(o[j], kinv[u]), 0x6240);
            }
            sm[wsw8 + kP8Threads * u] = make_uint4(w[0], w[1], w[2], w[3]);
            return;
        }
        const uint4 a = sm[(isv ? kP8V8 : kP8K8) + tid + kP8Threads * u];
        uint4* d = sm + (isv ? kP8V16 : kP8K16) + 32 * KS * u;
        uint32_t e0, o0, e1, o1, e2, o2, e3, o3;
        f8x4_to_f16(a.x, e0, o0);
        f8x4_to_f16(a.y, e1, o1);
        f8x4_to_f16(a.z, e2, o2);
        f8x4_to_f16(a.w, e3, o3);
        d[wsw0] = make_uint4(e0, o0, e1, o1);
        d[wsw1] = make_uint4(e2, o2, e3, o3);
    };
    // ldmatrix addresses (bytes): Q K^T B fragments of key pair-tile jp, k-step kk at
    // kq[kk & 3] + jp * 8192 + (kk >> 2) * 128; P V fragments of dim pair-tile jp, key step ks at
    // vq[jp & 3] + ks * 8192 + (jp >> 2) * 128
    const int mi = lane >> 3, L7 = lane & 7;
    unsigned kq[4], vq[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        kq[i] = QK8 ? sbase + 16u * (((mi >> 1) * 8 + L7) * 16 + ((2 * i + (mi & 1)) ^ L7))
                    : sbase + 16u * (kP8K16 + ((mi >> 1) * 8 + L7) * 32 + ((2 * i + (mi & 1)) ^ L7));
        vq[i] = sbase + 16u * (kP8V16 + ((mi & 1) * 8 + L7) * 32 + ((2 * i + (mi >> 1)) ^ L7));
    }

    load(false, 0);
    load(true, 0);
    cp_commit();
    cp_wait<0>();
    if (QK8) kscales(0);
#pragma unroll
    for (int u = 0; u < 4; u++) widen(false, u);
    if (ntiles > 1) load(false, 1);  // the K stage is free once K(0) is widened (own chunks)
    cp_commit();
    __syncthreads();

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int r0 = warp * 16 + g;
    const int qpos0 = p0 + q0 + min(r0, G - 1), qpos1 = p0 + q0 + min(r0 + 8, G - 1);
    const int rl0 = pm_rank(pm, p0 + q0);  // keys below are visible to every row of the block
    const float sq0 = QK8 ? s_sq[r0] : 1.f, sq1 = QK8 ? s_sq[r0 + 8] : 1.f;
    const float cq0 = smooth ? s_cq[r0] : 0.f, cq1 = smooth ? s_cq[r0 + 8] : 0.f;
    const float vscale = s.vs * 256.f;
    constexpr uint32_t kOnes = 0x3C003C00u;

    for (int tile = 0; tile < ntiles; tile++) {
        const int k0 = kbeg + tile * kP8Keys;
        // ---- phase A: S = Q K^T (fp16 accumulate) + widen V(tile), copy V(tile + 1)
        cp_wait<1>();
        uint32_t sc[NJ][2];
        float sf[QK8 ? NJ : 1][4];  // QK8: fp32 scores
        if (QK8) {
            uint32_t si[NJ][4];  // int32 accumulators, initialized to the bits of 1.5 * 2^23 (int -> float for free)
#pragma unroll
            for (int j = 0; j < NJ; j++) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0x4B400000u;
#pragma unroll
            for (int kk = 0; kk < D / 32; kk++) {
                if ((kk & 1) == 0) widen(true, kk >> 1);
#pragma unroll
                for (int jp = 0; jp < kP8Keys / 16; jp++) {
                    uint32_t b[4];
                    ldsm_x4a(b, kq[kk & 3] + jp * 4096 + (kk >> 2) * 128);
                    imma(si[jp * 2], qf[kk], b[0], b[1]);
                    imma(si[jp * 2 + 1], qf[kk], b[2], b[3]);
                }
            }
            const float4* skp = reinterpret_cast<const float4*>(sm + kP8SK);
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const float4 k4 = skp[j * 4 + t];  // keys 8j + 2t, 8j + 2t + 1: (scale, -1.5 * 2^23 * scale)
                sf[j][0] = fmaf(__uint_as_float(si[j][0]), k4.x, k4.y);
                sf[j][1] = fmaf(__uint_as_float(si[j][1]), k4.z, k4.w);
                sf[j][2] = fmaf(__uint_as_float(si[j][2]), k4.x, k4.y);
                sf[j][3] = fmaf(__uint_as_float(si[j][3]), k4.z, k4.w);
            }
        } else {
#pragma unroll
        for (int j = 0; j < NJ; j++) sc[j][0] = sc[j][1] = 0u;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            if (!QE_PF8_EARLY && (kk & 3) == 0) widen(true, kk >> 2);
            if (QE_PF8_EARLY && kk == 0) {
#pragma unroll
                for (int u = 0; u < 4; u++) widen(true, u);
                if (tile + 1 < ntiles) load(true, tile + 1);
            }
#pragma unroll
            for (int jp = 0; jp < kP8Keys / 16; jp++) {
                uint32_t b[4];
                ldsm_x4a(b, kq[kk & 3] + jp * 8192 + (kk >> 2) * 128);
                if (QE_PF8_ABL & 2) { sc[jp * 2][0] ^= b[0] ^ qf[kk][0]; sc[jp * 2 + 1][1] ^= b[3] ^ qf[kk][1]; continue; }
                mma_hacc(sc[jp * 2], qf[kk], b[0], b[1]);
                mma_hacc(sc[jp * 2 + 1], qf[kk], b[2], b[3]);
            }
        }
        }
        if (!QE_PF8_EARLY && tile + 1 < ntiles) load(true, tile + 1);
        cp_commit();
        float alpha[2];
        float rsum[2] = {0.f, 0.f};
        if (QK8 && (QE_PF8_ABL & 64)) {  // ablation: no softmax
            alpha[0] = alpha[1] = 1.f;
#pragma unroll
            for (int j = 0; j < NJ; j++) { sc[j][0] = pack_f16(sf[j][0], sf[j][1]); sc[j][1] = pack_f16(sf[j][2], sf[j][3]); }
        } else if (QK8) {  // fp32 softmax of the exact int8 scores; P -> fp16 A fragments in sc
            if (k0 + kP8Keys > min(kend, rl0)) {
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = k0 + j * 8 + 2 * t + (e & 1);
                        if (key >= kend || pm_unrank(pm, key) > ((e < 2) ? qpos0 : qpos1)) sf[j][e] = -INFINITY;
                    }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                float tmax = -INFINITY;  // of the scores without the row's Q scale (> 0)
#pragma unroll
                for (int j = 0; j < NJ; j++) tmax = fmaxf(tmax, fmaxf(sf[j][2 * r], sf[j][2 * r + 1]));
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
                const float qs = r ? sq1 : sq0, cq = r ? cq1 : cq0;
                const float nm = fmaxf(mx[r], fmaf(tmax, qs, cq));
                alpha[r] = mx[r] == -INFINITY ? 0.f : exp2f(mx[r] - nm);
                mx[r] = nm;
                const float mm = nm == -INFINITY ? 0.f : nm;
#pragma unroll
                for (int j = 0; j < NJ; j++)
                    sc[j][r] = pack_f16(exp2f(fmaf(sf[j][2 * r], qs, cq - mm)), exp2f(fmaf(sf[j][2 * r + 1], qs, cq - mm)));
            }
        } else {
        // ---- softmax (fp16): mask only the tiles reaching the block's own queries / the range end
        if (k0 + kP8Keys > min(kend, rl0)) {
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int qp = r ? qpos1 : qpos0;
                    const int key = k0 + j * 8 + 2 * t;
                    const bool m0 = key >= kend || pm_unrank(pm, key) > qp;
                    const bool m1 = key + 1 >= kend || pm_unrank(pm, key + 1) > qp;
                    sc[j][r] = (m0 ? 0xFC00u : (sc[j][r] & 0xFFFFu)) | (m1 ? 0xFC000000u : (sc[j][r] & 0xFFFF0000u));
                }
        }
        uint32_t mh2[2];
        if (QE_PF8_ABL & 16) { alpha[0] = alpha[1] = 1.f; mx[0] = mx[1] = 0.f; } else
#pragma unroll
        for (int r = 0; r < 2; r++) {
            uint32_t hm = sc[0][r];
#pragma unroll
            for (int j = 1; j < NJ; j++) hm = h2_max(hm, sc[j][r]);
            float tmax = h2_maxf(hm);
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float nm = fmaxf(mx[r], tmax);
            alpha[r] = mx[r] == -INFINITY ? 0.f : exp2f(mx[r] - nm);
            mx[r] = nm;
            const __half mh = __float2half_rn(nm == -INFINITY ? 0.f : nm);  // exact: an fp16 score
            const __half2 m2 = __halves2half2(mh, mh);
            mh2[r] = *reinterpret_cast<const uint32_t*>(&m2);
        }
        if (QE_PF8_DBG & 1) {
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const float2 f = __half22float2(*reinterpret_cast<__half2*>(&sc[j][r]));
                    const float mm = mx[r] == -INFINITY ? 0.f : mx[r];
                    const float pa = exp2f(f.x - mm), pb = exp2f(f.y - mm);
                    rsum[r] += pa + pb;
                    sc[j][r] = pack_f16(pa, pb);
                }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 1);
                rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 2);
            }
        } else if (!(QE_PF8_ABL & 16)) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            sc[j][0] = h2_exp2(h2_sub(sc[j][0], mh2[0]));
            sc[j][1] = h2_exp2(h2_sub(sc[j][1], mh2[1]));
        }
        }
        }  // !QK8
        __syncthreads();  // V(tile) widened by all; Q K^T(tile) done everywhere -> K buffer free

        // ---- phase B: O = O * alpha + P V (fp16 accumulate over the tile) + widen K(tile + 1)
        cp_wait<1>();
        {
            uint32_t cl[2] = {0u, 0u};
#pragma unroll
            for (int ks = 0; ks < kP8Keys / 16; ks++) {
                const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                mma_hacc(cl, a, kOnes, kOnes);
            }
            if ((QE_PF8_DBG & 1) && !QK8) {
                l[0] = l[0] * alpha[0] + rsum[0];
                l[1] = l[1] * alpha[1] + rsum[1];
            } else {
            l[0] = l[0] * alpha[0] + __low2float(*reinterpret_cast<__half2*>(&cl[0]));
            l[1] = l[1] * alpha[1] + __low2float(*reinterpret_cast<__half2*>(&cl[1]));
            }
        }
        const bool wk = tile + 1 < ntiles;
        if (QK8 && wk && !(QE_PF8_ABL & 1)) kscales(tile + 1);
#pragma unroll
        for (int jp = 0; jp < 16; jp++) {
            if (!QE_PF8_EARLY && (jp & 3) == 0 && wk) widen(false, jp >> 2);
            if (QE_PF8_EARLY && jp == 0) {
                if (wk) {
#pragma unroll
                    for (int u = 0; u < 4; u++) widen(false, u);
                }
                if (tile + 2 < ntiles) load(false, tile + 2);
            }
            uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u};
#ifndef QE_PF8_PV32  // 1 (default): P V accumulated in fp16 over 32 keys (two halves added in fp32); 0: over the whole 64-key tile (2-4 % faster, ~15-25 % higher KL vs fp32 attention)
#define QE_PF8_PV32 1
#endif
            uint32_t d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
            for (int ks = 0; ks < kP8Keys / 16; ks++) {
                const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                uint32_t b[4];
                ldsm_x4ta(b, vq[jp & 3] + ks * 8192 + (jp >> 2) * 128);
                if (QE_PF8_ABL & 4) { c0[0] ^= b[0] ^ a[0]; c1[1] ^= b[3] ^ a[1]; continue; }
                if (QE_PF8_PV32 && ks >= 2) {
                    mma_hacc(d0, a, b[0], b[1]);
                    mma_hacc(d1, a, b[2], b[3]);
                } else {
                    mma_hacc(c0, a, b[0], b[1]);
                    mma_hacc(c1, a, b[2], b[3]);
                }
            }
            if (QE_PF8_PV32) {  // c += d in fp32, written back as the fp32 sums below
                const float2 a0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0])), b0 = __half22float2(*reinterpret_cast<__half2*>(&d0[0]));
                const float2 a1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1])), b1 = __half22float2(*reinterpret_cast<__half2*>(&d0[1]));
                const float2 a2 = __half22float2(*reinterpret_cast<__half2*>(&c1[0])), b2 = __half22float2(*reinterpret_cast<__half2*>(&d1[0]));
                const float2 a3 = __half22float2(*reinterpret_cast<__half2*>(&c1[1])), b3 = __half22float2(*reinterpret_cast<__half2*>(&d1[1]));
                float* pa = o[jp * 2];
                float* pb = o[jp * 2 + 1];
                pa[0] = pa[0] * alpha[0] + (a0.x + b0.x); pa[1] = pa[1] * alpha[0] + (a0.y + b0.y);
                pa[2] = pa[2] * alpha[1] + (a1.x + b1.x); pa[3] = pa[3] * alpha[1] + (a1.y + b1.y);
                pb[0] = pb[0] * alpha[0] + (a2.x + b2.x); pb[1] = pb[1] * alpha[0] + (a2.y + b2.y);
                pb[2] = pb[2] * alpha[1] + (a3.x + b3.x); pb[3] = pb[3] * alpha[1] + (a3.y + b3.y);
                continue;
            }
            if (QE_PF8_ABL & 32) {
                o[jp * 2][0] += __int_as_float(c0[0]); o[jp * 2][2] += __int_as_float(c0[1]);
                o[jp * 2 + 1][0] += __int_as_float(c1[0]); o[jp * 2 + 1][2] += __int_as_float(c1[1]);
                continue;
            }
            const float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0]));
            const float2 x1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1]));
            const float2 y0 = __half22float2(*reinterpret_cast<__half2*>(&c1[0]));
            const float2 y1 = __half22float2(*reinterpret_cast<__half2*>(&c1[1]));
            float* pa = o[jp * 2];
            float* pb = o[jp * 2 + 1];
            pa[0] = pa[0] * alpha[0] + x0.x; pa[1] = pa[1] * alpha[0] + x0.y;
            pa[2] = pa[2] * alpha[1] + x1.x; pa[3] = pa[3] * alpha[1] + x1.y;
            pb[0] = pb[0] * alpha[0] + y0.x; pb[1] = pb[1] * alpha[0] + y0.y;
            pb[2] = pb[2] * alpha[1] + y1.x; pb[3] = pb[3] * alpha[1] + y1.y;
        }
        if (!QE_PF8_EARLY && tile + 2 < ntiles) load(false, tile + 2);
        cp_commit();
        __syncthreads();  // K(tile + 1) widened by all; P V(tile) done everywhere -> V buffer free
    }
    cp_wait<0>();

    // ---- output: positions 8j + 2t + {0,1} hold dims (t even) 8j + 2t + {0, 2} / (t odd) 8j + 2t - 1 + {0, 2};
    // a lane-pair exchange restores consecutive pairs: even lane -> dims base + {0,1}, odd -> base + {2,3}
    const bool odd = t & 1;
    const int dbase = 4 * (t >> 1) + (odd ? 2 : 0);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = r0 + 8 * r;
        float v[32][2];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const float x0 = o[j][2 * r], x1 = o[j][2 * r + 1];
            const float rcv = __shfl_xor_sync(0xffffffffu, odd ? x0 : x1, 1);
            v[j][0] = odd ? rcv : x0;
            v[j][1] = odd ? x1 : rcv;
        }
        if (row >= G) continue;
        if (part) {
#pragma unroll
            for (int j = 0; j < 32; j++)
                *reinterpret_cast<float2*>(pacc(row) + j * 8 + dbase) = make_float2(v[j][0] * vscale, v[j][1] * vscale);
            if (t == 0) {
                pml(row)[0] = mx[r] == -INFINITY ? -FLT_MAX : mx[r] * 0.6931471805599453f;
                pml(row)[1] = l[r];
            }
        } else {
            const int qi = q0 + row;
            const int no = s.no ? s.no : s.nq;
            if (h < s.ob || h >= s.ob + no) continue;
            const float inv = vscale / l[r];
            const float* gate = proj + (size_t)qi * ld + h * 512 + 256;
            bf16* op = out + ((size_t)qi * no + (h - s.ob)) * D;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int d = j * 8 + dbase;
                const float g0 = 1.f / (1.f + __expf(-gate[d]));
                const float g1 = 1.f / (1.f + __expf(-gate[d + 1]));
                *reinterpret_cast<bf162*>(op + d) = __floats2bfloat162_rn(v[j][0] * inv * g0, v[j][1] * inv * g1);
            }
        }
    }
}

// ---------------------------------------------------------------- fp8 prefill attention, ping-pong
// attn_pp8_kernel (QE_PATTN_PP=1): like attn_pf8_kernel<false, 8> (128 rows, 8 warps, fp8 tiles widened once
// for all warps), but 32-key tiles with double-buffered fp16 K and V (80 KB) and two warp groups (warps 0-3:
// rows 0-63, warps 4-7: rows 64-127) one slot apart: in slot 2t + g the group g computes Q K^T + softmax of
// tile t, in slot 2t + g + 1 its P V. The two warps of a scheduler are then always in different phases, so
// one group's softmax / epilogue overlaps the other's MMAs instead of every warp idling the tensor pipe at
// the same time. All threads widen in every slot (even slots K(t+1), odd slots V(t+1)), one barrier per slot.
constexpr int kPPRows = 128, kPPKeys = 32, kPPThreads = 256;
constexpr int kPPK16 = 0, kPPV16 = 2048, kPPK8 = 4096, kPPV8 = 4608;  // uint4 offsets; K16 / V16: 2 buffers of 1024
constexpr size_t kPPSmem = (size_t)5120 * 16;                       // 80 KB

__global__ void __launch_bounds__(kPPThreads, 1) attn_pp8_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    AttnShape s, const float* __restrict__ proj, int ld, const int* __restrict__ pos, int m,
    bf16* __restrict__ out, float* __restrict__ work) {
    extern __shared__ __align__(128) uint4 sm[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0];
    const int QT = (m + kPPRows - 1) / kPPRows, S = s.pf_splits;
    const int qt = blockIdx.x % QT, sp = (blockIdx.x / QT) % S, hh = blockIdx.x / (QT * S);
    int h = 0;
    {
        int kvo[kMaxLocalKv];
        for (int j = 0; j < s.nkv; j++) kvo[j] = j;
        for (int a = 1; a < s.nkv; a++)
            for (int b = a; b > 0; b--) {
                const PosMap& x = s.kv_pm[kvo[b]];
                const PosMap& y = s.kv_pm[kvo[b - 1]];
                if ((long long)x.A * y.P > (long long)y.A * x.P) { const int tt = kvo[b]; kvo[b] = kvo[b - 1]; kvo[b - 1] = tt; }
                else break;
            }
        int rem = hh;
        for (int j = 0; j < s.nkv; j++) {
            const int kj = kvo[j], n = s.kv_qe[kj] - s.kv_qb[kj];
            if (rem < n) { h = s.kv_qb[kj] + rem; break; }
            rem -= n;
        }
    }
    const int q0 = qt * kPPRows, kv = s.q2kv[h];
    const int G = min(kPPRows, m - q0);
    const PosMap pm = s.kv_pm[kv];
    const int splitl = pm_rank(pm, s.split);
    int kbeg = 0, kend = pm_rank(pm, p0 + min(q0 + kPPRows, m));
    const int split = sp;
    if (S > 1) {
        int chunk = (kend + S - 1) / S;
        chunk = (chunk + kPPKeys - 1) / kPPKeys * kPPKeys;
        kbeg = split * chunk;
        kend = min(kend, kbeg + chunk);
    }
    const bool part = S > 1 || s.kv_x[kv] != 0;
    auto pacc = [&](int row) { return work + (((size_t)split * m + q0 + row) * s.nq + h) * 256; };
    auto pml = [&](int row) {
        return work + (size_t)S * m * s.nq * 256 + (((size_t)split * m + q0 + row) * s.nq + h) * 2;
    };
    if (part && kbeg >= kend) {
        for (int r = tid; r < G; r += kPPThreads) {
            pml(r)[0] = -FLT_MAX;
            pml(r)[1] = 0.f;
        }
        return;
    }
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    {
        const float sc = 0.0625f * 1.4426950408889634f * s.ks * 256.f;
        for (int i = tid; i < kPPRows * 32; i += kPPThreads) {
            const int r = i >> 5, ch = i & 31;
            const int qi = q0 + min(r, G - 1);
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(q + ((size_t)qi * s.nq + h) * D) + ch);
            const float2 a0 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.x));
            const float2 a1 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.y));
            const float2 a2 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.z));
            const float2 a3 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&v.w));
            sm[sw(r, ch)] = make_uint4(pack_f16(a0.x * sc, a1.x * sc), pack_f16(a0.y * sc, a1.y * sc),
                                       pack_f16(a2.x * sc, a3.x * sc), pack_f16(a2.y * sc, a3.y * sc));
        }
    }
    __syncthreads();
    uint32_t qf[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) ldsm_x4(qf[kk], &sm[sw(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))]);
    __syncthreads();

    const int cr = tid >> 4, cc = tid & 15;  // chunk (key cr + 16 u, column cc), u = 0, 1
    const uint8_t* kb = kc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* vb = vc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* kb2 = (const uint8_t*)s.kc2 + head_rows(s, kv, s.split) * D;
    const uint8_t* vb2 = (const uint8_t*)s.vc2 + head_rows(s, kv, s.split) * D;
    const int ntiles = (kend - kbeg + kPPKeys - 1) / kPPKeys;
    auto load = [&](bool isv, int tile) {
        const int k0 = kbeg + tile * kPPKeys;
        uint4* dst = sm + (isv ? kPPV8 : kPPK8) + tid;
        const uint8_t* own = isv ? vb : kb;
        const uint8_t* pre = isv ? vb2 : kb2;
        if (k0 + kPPKeys <= kend && (k0 >= splitl || k0 + kPPKeys <= splitl)) {
            const uint8_t* src = (k0 >= splitl ? own + (size_t)(k0 - splitl) * D : pre + (size_t)k0 * D) +
                                 (size_t)cr * D + cc * 16;
#pragma unroll
            for (int u = 0; u < 2; u++) cp_async16m(dst + 256 * u, src + (size_t)u * 16 * D, true);
        } else {
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int key = k0 + cr + 16 * u;
                const int kk = min(key, kend - 1);
                const uint8_t* src = (kk < splitl ? pre + (size_t)kk * D : own + (size_t)(kk - splitl) * D) + cc * 16;
                cp_async16m(dst + 256 * u, src, key < kend);
            }
        }
    };
    const int wsw0 = cr * 32 + ((2 * cc) ^ (cr & 7)), wsw1 = cr * 32 + ((2 * cc + 1) ^ (cr & 7));
    auto widen = [&](bool isv, int buf) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const uint4 a = sm[(isv ? kPPV8 : kPPK8) + tid + 256 * u];
            uint4* d = sm + (isv ? kPPV16 : kPPK16) + buf * 1024 + 512 * u;
            uint32_t e0, o0, e1, o1, e2, o2, e3, o3;
            f8x4_to_f16(a.x, e0, o0);
            f8x4_to_f16(a.y, e1, o1);
            f8x4_to_f16(a.z, e2, o2);
            f8x4_to_f16(a.w, e3, o3);
            d[wsw0] = make_uint4(e0, o0, e1, o1);
            d[wsw1] = make_uint4(e2, o2, e3, o3);
        }
    };
    const int mi = lane >> 3, L7 = lane & 7;
    unsigned kq[4], vq[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        kq[i] = sbase + 16u * (kPPK16 + ((mi >> 1) * 8 + L7) * 32 + ((2 * i + (mi & 1)) ^ L7));
        vq[i] = sbase + 16u * (kPPV16 + ((mi & 1) * 8 + L7) * 32 + ((2 * i + (mi >> 1)) ^ L7));
    }

    load(false, 0);
    load(true, 0);
    cp_commit();
    cp_wait<0>();
    widen(false, 0);
    widen(true, 0);
    if (ntiles > 1) load(false, 1);
    cp_commit();
    if (ntiles > 1) load(true, 1);
    cp_commit();
    __syncthreads();

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int r0 = warp * 16 + g;
    const int qpos0 = p0 + q0 + min(r0, G - 1), qpos1 = p0 + q0 + min(r0 + 8, G - 1);
    const int rl0 = pm_rank(pm, p0 + q0);
    const float vscale = s.vs * 256.f;
    constexpr uint32_t kOnes = 0x3C003C00u;
    const int grp = warp >> 2;
    uint32_t sc[4][2];  // S, then P of the group's current tile (carried from its Q K^T slot to its P V slot)
    float alpha[2] = {1.f, 1.f};

    for (int sl = 0; sl < 2 * ntiles + 1; sl++) {
        // ---- shared pipeline work: even slot 2t widens K(t+1), odd slot 2t+1 widens V(t+1); copies 2 slots ahead
        cp_wait<1>();
        {
            const int tw = (sl >> 1) + 1;
            const bool isv = sl & 1;
            if (tw < ntiles) widen(isv, tw & 1);
            if (tw + 1 < ntiles) load(isv, tw + 1);
            cp_commit();
        }
        const int ls = sl - grp;
        if (ls >= 0 && ls < 2 * ntiles) {
            const int tile = ls >> 1, b = tile & 1;
            const int k0 = kbeg + tile * kPPKeys;
            if (!(ls & 1)) {  // Q K^T + softmax of tile
#pragma unroll
                for (int j = 0; j < 4; j++) sc[j][0] = sc[j][1] = 0u;
#pragma unroll
                for (int kk = 0; kk < D / 16; kk++)
#pragma unroll
                    for (int jp = 0; jp < 2; jp++) {
                        uint32_t bb[4];
                        ldsm_x4a(bb, kq[kk & 3] + b * 16384 + jp * 8192 + (kk >> 2) * 128);
                        mma_hacc(sc[jp * 2], qf[kk], bb[0], bb[1]);
                        mma_hacc(sc[jp * 2 + 1], qf[kk], bb[2], bb[3]);
                    }
                if (k0 + kPPKeys > min(kend, rl0)) {
#pragma unroll
                    for (int j = 0; j < 4; j++)
#pragma unroll
                        for (int r = 0; r < 2; r++) {
                            const int qp = r ? qpos1 : qpos0;
                            const int key = k0 + j * 8 + 2 * t;
                            const bool m0 = key >= kend || pm_unrank(pm, key) > qp;
                            const bool m1 = key + 1 >= kend || pm_unrank(pm, key + 1) > qp;
                            sc[j][r] = (m0 ? 0xFC00u : (sc[j][r] & 0xFFFFu)) | (m1 ? 0xFC000000u : (sc[j][r] & 0xFFFF0000u));
                        }
                }
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    uint32_t hm = h2_max(h2_max(sc[0][r], sc[1][r]), h2_max(sc[2][r], sc[3][r]));
                    float tmax = h2_maxf(hm);
                    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
                    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
                    const float nm = fmaxf(mx[r], tmax);
                    alpha[r] = mx[r] == -INFINITY ? 0.f : exp2f(mx[r] - nm);
                    mx[r] = nm;
                    const __half mh = __float2half_rn(nm == -INFINITY ? 0.f : nm);
                    const __half2 m2 = __halves2half2(mh, mh);
                    const uint32_t mh2 = *reinterpret_cast<const uint32_t*>(&m2);
#pragma unroll
                    for (int j = 0; j < 4; j++) sc[j][r] = h2_exp2(h2_sub(sc[j][r], mh2));
                }
            } else {  // O = O * alpha + P V of tile
                {
                    uint32_t cl[2] = {0u, 0u};
#pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                        mma_hacc(cl, a, kOnes, kOnes);
                    }
                    l[0] = l[0] * alpha[0] + __low2float(*reinterpret_cast<__half2*>(&cl[0]));
                    l[1] = l[1] * alpha[1] + __low2float(*reinterpret_cast<__half2*>(&cl[1]));
                }
#pragma unroll
                for (int jp = 0; jp < 16; jp++) {
                    uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u};
#pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                        uint32_t bb[4];
                        ldsm_x4ta(bb, vq[jp & 3] + b * 16384 + ks * 8192 + (jp >> 2) * 128);
                        mma_hacc(c0, a, bb[0], bb[1]);
                        mma_hacc(c1, a, bb[2], bb[3]);
                    }
                    const float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0]));
                    const float2 x1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1]));
                    const float2 y0 = __half22float2(*reinterpret_cast<__half2*>(&c1[0]));
                    const float2 y1 = __half22float2(*reinterpret_cast<__half2*>(&c1[1]));
                    float* pa = o[jp * 2];
                    float* pb = o[jp * 2 + 1];
                    pa[0] = pa[0] * alpha[0] + x0.x; pa[1] = pa[1] * alpha[0] + x0.y;
                    pa[2] = pa[2] * alpha[1] + x1.x; pa[3] = pa[3] * alpha[1] + x1.y;
                    pb[0] = pb[0] * alpha[0] + y0.x; pb[1] = pb[1] * alpha[0] + y0.y;
                    pb[2] = pb[2] * alpha[1] + y1.x; pb[3] = pb[3] * alpha[1] + y1.y;
                }
            }
        }
        __syncthreads();
    }
    cp_wait<0>();

    const bool odd = t & 1;
    const int dbase = 4 * (t >> 1) + (odd ? 2 : 0);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = r0 + 8 * r;
        float v[32][2];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const float x0 = o[j][2 * r], x1 = o[j][2 * r + 1];
            const float rcv = __shfl_xor_sync(0xffffffffu, odd ? x0 : x1, 1);
            v[j][0] = odd ? rcv : x0;
            v[j][1] = odd ? x1 : rcv;
        }
        if (row >= G) continue;
        if (part) {
#pragma unroll
            for (int j = 0; j < 32; j++)
                *reinterpret_cast<float2*>(pacc(row) + j * 8 + dbase) = make_float2(v[j][0] * vscale, v[j][1] * vscale);
            if (t == 0) {
                pml(row)[0] = mx[r] == -INFINITY ? -FLT_MAX : mx[r] * 0.6931471805599453f;
                pml(row)[1] = l[r];
            }
        } else {
            const int qi = q0 + row;
            const int no = s.no ? s.no : s.nq;
            if (h < s.ob || h >= s.ob + no) continue;
            const float inv = vscale / l[r];
            const float* gate = proj + (size_t)qi * ld + h * 512 + 256;
            bf16* op = out + ((size_t)qi * no + (h - s.ob)) * D;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int d = j * 8 + dbase;
                const float g0 = 1.f / (1.f + __expf(-gate[d]));
                const float g1 = 1.f / (1.f + __expf(-gate[d + 1]));
                *reinterpret_cast<bf162*>(op + d) = __floats2bfloat162_rn(v[j][0] * inv * g0, v[j][1] * inv * g1);
            }
        }
    }
}

// ---------------------------------------------------------------- fp8 decode attention
// Decode / verification attention over an FP8 cache. Block = one (kv head, split); KG key groups
// stream alternate 16-key tiles of the split through their own double-buffered cp.async stage,
// each with RT warps (16 rows each) and its own online-softmax state; the groups merge at the
// end into the same partial format as attn_tc_kernel (so attn_combine is unchanged).
// K fragments come straight from the fp8 tile: one u32 holds dims 16kk + 4t .. +3, used as the
// mma k-indices (2t, 2t+1, 2t+8, 2t+9); Q is loaded with the same dim permutation, which leaves
// q.k unchanged.
constexpr int kDecTile = 16;
constexpr int kDecStages = 3;
template <int RT, int KG>
constexpr size_t dec8_smem() {
    const size_t stage = (size_t)KG * kDecStages * 2 * kDecTile * D;   // [KG][stage][K|V][16][256]
    const size_t merge = (size_t)(KG - 1) * RT * 32 * (32 * 4 + 4) * 4;  // o + (mx, l) per lane
    return stage > merge ? stage : merge;
}

template <int RT, int KG>
__device__ __forceinline__ void attn_dec8_dev(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const AttnShape& s, const SeqView& v, const int* __restrict__ pos, int m, float* __restrict__ work) {
    extern __shared__ __align__(128) uint4 sm[];
    uint8_t* smb = reinterpret_cast<uint8_t*>(sm);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kg = warp / RT, rt = warp % RT;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0];
    const int kv = blockIdx.x, split = blockIdx.y;
    const int G = (s.kv_qe[kv] - s.kv_qb[kv]) * m;
    // local key space of this GPU's share of the head (kv-pos-split; a full head: rank = id)
    const PosMap pm = s.kv_pm[kv];
    const int p0l = pm_rank(pm, p0), splitl = pm_rank(pm, v.split);
    const int L = pm_rank(pm, p0 + m);
    const int k0wa = dec8_first(p0, s.win), k0w = pm_rank(pm, k0wa);  // sliding window (MTP drafts)
    const int chunk = dec8_chunk_pm(s, kv, p0l - k0w, p0 - k0wa);
    const int kbeg = k0w + split * chunk, kend = min(L, kbeg + chunk);
    float* wacc = work + ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 256;
    float* wml = work + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256 +
                 ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 2;
    if (kbeg >= kend) {
        for (int r = tid; r < G; r += blockDim.x) {
            wml[r * 2] = -FLT_MAX;
            wml[r * 2 + 1] = 0.f;
        }
        return;
    }
    // rows of this warp: r = rt * 16 + g (+8) -> (head in group, token)
    auto row_q = [&](int r, int& qrow, int& qpos) {
        const int rr = min(r, G - 1);
        const int hl = rr / m, mi = rr % m;
        qrow = mi * s.nq + s.kv_qb[kv] + hl;
        qpos = p0 + mi;
    };
    const int r0 = rt * 16 + g, r1 = r0 + 8;
    int qrow0, qrow1, qpos0, qpos1;
    row_q(r0, qrow0, qpos0);
    row_q(r1, qrow1, qpos1);
    // Q fragments (fp16), dims permuted as described above
    uint32_t qf[D / 16][4];
    {
        const bf16* qa = q + (size_t)qrow0 * D;
        const bf16* qb = q + (size_t)qrow1 * D;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            const int d = kk * 16 + 4 * t;
            auto cv = [](const bf16* p) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(p));
                return pack_f16(f.x, f.y);
            };
            qf[kk][0] = cv(qa + d);
            qf[kk][1] = cv(qb + d);
            qf[kk][2] = cv(qa + d + 2);
            qf[kk][3] = cv(qb + d + 2);
        }
    }
    // this group's stages: [2][K|V][16 keys][256 B]; K chunk c of key r at (c ^ (r & 7)),
    // V chunk c of key r at (c ^ ((r >> 1) & 7)) (conflict-free fragment reads)
    uint8_t* gs = smb + (size_t)kg * kDecStages * 2 * kDecTile * D;
    const int ntiles = (kend - kbeg + kDecTile - 1) / kDecTile;
    const int gthreads = RT * 32, gtid = rt * 32 + lane;
    const uint8_t* kcb = kc + head_rows(s, kv, v.max_t) * D;
    const uint8_t* vcb = vc + head_rows(s, kv, v.max_t) * D;
    const uint8_t* kcb2 = (const uint8_t*)v.kc2 + head_rows(s, kv, v.split) * D;
    const uint8_t* vcb2 = (const uint8_t*)v.vc2 + head_rows(s, kv, v.split) * D;
    auto load_tile = [&](int stage, int tile) {
        const int k0 = kbeg + tile * kDecTile;
        uint8_t* ks = gs + (size_t)stage * 2 * kDecTile * D;
        uint8_t* vs = ks + kDecTile * D;
        for (int i = gtid; i < kDecTile * 16; i += gthreads) {
            const int r = i >> 4, c = i & 15;
            const int key = k0 + r;
            const bool ok = key < kend;
            const int kk = min(key, kend - 1);
            const bool pre = kk < splitl;  // shared prefix region
            const size_t off = (size_t)(pre ? kk : kk - splitl) * D + c * 16;
            cp_async16(ks + r * D + ((c ^ (r & 7)) << 4), (pre ? kcb2 : kcb) + off, ok);
            cp_async16(vs + r * D + ((c ^ ((r >> 1) & 7)) << 4), (pre ? vcb2 : vcb) + off, ok);
        }
    };
    auto group_sync = [&]() {
        if (RT == 1) __syncwarp();
        else asm volatile("bar.sync %0, %1;" ::"r"(1 + kg), "r"(RT * 32));
    };
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float mx[2] = {-FLT_MAX, -FLT_MAX}, l[2] = {0.f, 0.f};
    const float kscale = s.ks * 256.f, vscale = s.vs * 256.f;
    const float scale = 0.0625f * 1.4426950408889634f * kscale;

    int it = 0;
#pragma unroll
    for (int sidx = 0; sidx < kDecStages - 1; sidx++) {
        if (kg + sidx * KG < ntiles) load_tile(sidx, kg + sidx * KG);
        cp_commit();
    }
    for (int tile = kg; tile < ntiles; tile += KG, it++) {
        const int nt = tile + (kDecStages - 1) * KG;
        if (nt < ntiles) load_tile((it + kDecStages - 1) % kDecStages, nt);
        cp_commit();
        cp_wait<kDecStages - 1>();
        group_sync();
        const uint8_t* ks = gs + (size_t)(it % kDecStages) * 2 * kDecTile * D;
        const uint8_t* vs = ks + kDecTile * D;
        // S = Q K^T (16 rows x 16 keys), summed as four 64-dim partials in a fixed order so the
        // result is bit-identical to attn_dec8d (which splits the dims across warps): a token's
        // attention does not depend on how many rows are verified with it
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int wg = 0; wg < 4; wg++) {
            float pp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                const int kk = wg * 4 + k4;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int r = j * 8 + g;
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(ks + r * D + ((kk ^ (r & 7)) << 4) + 4 * t);
                    mma_f16(pp[j], qf[kk], e4m3x2_to_f16x2(__byte_perm(w, 0, 0x4140)),
                            e4m3x2_to_f16x2(__byte_perm(w, 0, 0x4342)));
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) sc[j][e] += pp[j][e];
        }
        const int k0 = kbeg + tile * kDecTile;
        const bool tail = k0 + kDecTile > p0l;  // only these tiles need the absolute position
        float tmax[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int key = k0 + j * 8 + 2 * t + (e & 1);
                const int qp = (e < 2) ? qpos0 : qpos1;
                float v = sc[j][e] * scale;
                if (key >= kend || (tail && pm_unrank(pm, key) > qp)) v = -FLT_MAX;
                sc[j][e] = v;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], v);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
            const float nm = fmaxf(mx[r], tmax[r]);
            alpha[r] = (mx[r] == -FLT_MAX) ? 0.f : exp2f(mx[r] - nm);
            mx[r] = nm;
        }
        float rs[2] = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = sc[j][e];
                const float p = (v == -FLT_MAX) ? 0.f : exp2f(v - mx[e >> 1]);
                sc[j][e] = p;
                rs[e >> 1] += p;
            }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
            rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
            l[r] = l[r] * alpha[r] + rs[r];
        }
        uint32_t a[4];
        a[0] = pack_f16(sc[0][0], sc[0][1]);
        a[1] = pack_f16(sc[0][2], sc[0][3]);
        a[2] = pack_f16(sc[1][0], sc[1][1]);
        a[3] = pack_f16(sc[1][2], sc[1][3]);
        // O += P V: B fragment (keys 2t, 2t+1 | 2t+8, 2t+9; dim 8 nd + g) from fp8 bytes
        const int ka = 2 * t, kb2 = 2 * t + 8;
        const uint8_t* va0 = vs + ka * D;
        const uint8_t* va1 = vs + (ka + 1) * D;
        const uint8_t* vb0 = vs + kb2 * D;
        const uint8_t* vb1 = vs + (kb2 + 1) * D;
        const int sa = (ka >> 1) & 7, sb = (kb2 >> 1) & 7;  // keys 2t and 2t+1 share a swizzle
#pragma unroll
        for (int nd = 0; nd < 32; nd++) {
            const int d = nd * 8 + g, c = d >> 4, cb = d & 15;
            const int oa = ((c ^ sa) << 4) + cb, ob = ((c ^ sb) << 4) + cb;
            const uint32_t x0 = (uint32_t)va0[oa] | ((uint32_t)va1[oa] << 16);
            const uint32_t x1 = (uint32_t)vb0[ob] | ((uint32_t)vb1[ob] << 16);
            o[nd][0] *= alpha[0]; o[nd][1] *= alpha[0];
            o[nd][2] *= alpha[1]; o[nd][3] *= alpha[1];
            mma_f16(o[nd], a, e4m3x2_to_f16x2(x0), e4m3x2_to_f16x2(x1));
        }
        group_sync();  // the stage is reloaded next iteration
    }
    cp_wait<0>();
    // merge key groups: groups 1.. write (o, mx, l) to shared memory, group 0 accumulates
    __syncthreads();
    float* mb = reinterpret_cast<float*>(sm);
    if (kg > 0) {
        float* d = mb + ((size_t)((kg - 1) * RT + rt) * 32 + lane) * (32 * 4 + 4);
#pragma unroll
        for (int j = 0; j < 32; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) d[j * 4 + e] = o[j][e];
        d[128] = mx[0]; d[129] = mx[1]; d[130] = l[0]; d[131] = l[1];
    }
    __syncthreads();
    if (kg > 0) return;
#pragma unroll
    for (int og = 1; og < KG; og++) {
        const float* d = mb + ((size_t)((og - 1) * RT + rt) * 32 + lane) * (32 * 4 + 4);
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const float om = d[128 + r], ol = d[130 + r];
            const float nm = fmaxf(mx[r], om);
            const float fa = mx[r] == -FLT_MAX ? 0.f : exp2f(mx[r] - nm);
            const float fb = om == -FLT_MAX ? 0.f : exp2f(om - nm);
            l[r] = l[r] * fa + ol * fb;
            mx[r] = nm;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                o[j][r * 2] = o[j][r * 2] * fa + d[j * 4 + r * 2] * fb;
                o[j][r * 2 + 1] = o[j][r * 2 + 1] * fa + d[j * 4 + r * 2 + 1] * fb;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = r == 0 ? r0 : r1;
        if (row >= G) continue;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int d = j * 8 + 2 * t;
            *reinterpret_cast<float2*>(wacc + (size_t)row * 256 + d) =
                make_float2(o[j][r * 2] * vscale, o[j][r * 2 + 1] * vscale);
        }
        if (t == 0) {
            wml[row * 2] = mx[r] == -FLT_MAX ? -FLT_MAX : mx[r] * 0.6931471805599453f;
            wml[row * 2 + 1] = l[r];
        }
    }
}

// fp8 decode attention for large query groups (G > 32 rows: e.g. 6 heads x 7-8 verified tokens).
// 4 warps split the 256 head dims (64 each): partial Q.K^T over its dims for all MT row tiles
// (each K fragment converted once, reused by every row tile), exchanged through shared memory
// for the full scores; softmax computed (identically) in every warp; P.V for its own 64 dims.
// Fewer registers than attn_dec8 (two blocks per SM) and no per-warp duplicate fp8 conversion.
#ifndef QE_D8TILE
#define QE_D8TILE 32
#endif
#ifndef QE_D8STAGES
#define QE_D8STAGES 2
#endif
// kD8Tile keys per tile: P.V partials of a whole tile accumulate in fp16 before the fp32 update
constexpr int kD8Warps = 4, kD8Stages = QE_D8STAGES, kD8Tile = QE_D8TILE, kD8J = kD8Tile / 8;
template <int MT>
constexpr size_t dec8d_smem() {
    // stages, partial scores (fp16, exactly the MMA results), per row tile P fragments + alpha
    return (size_t)kD8Stages * 2 * kD8Tile * D + (size_t)kD8Warps * MT * kD8J * 32 * 8 +
           (size_t)MT * 32 * (kD8J * 2 + 4) * 4;
}

template <int MT>
__device__ __forceinline__ void attn_dec8d_dev(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const AttnShape& s, const SeqView& v, const int* __restrict__ pos, int m, float* __restrict__ work,
    int split, int rt0, int rs) {
    extern __shared__ __align__(128) uint4 sm[];
    uint8_t* stg = reinterpret_cast<uint8_t*>(sm);                       // [stage][K|V][16][256]
    uint2* sx = reinterpret_cast<uint2*>(stg + (size_t)kD8Stages * 2 * kD8Tile * D);  // partial S
    constexpr int PW = kD8J * 2 + 4;  // words per lane: P fragments, alpha (16-byte aligned)
    uint32_t* spa = reinterpret_cast<uint32_t*>(sx + (size_t)kD8Warps * MT * kD8J * 32);  // [MT][32][PW]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0];
    const int kv = blockIdx.x;
    const int G = (s.kv_qe[kv] - s.kv_qb[kv]) * m;
    if (rt0 * 16 >= G) return;
    if (rs) {
        const bool short_ctx = pos[0] + m + 8 <= kD8RsMax;
        if ((rs == 1) != short_ctx) return;
    }
    // local key space of this GPU's share of the head (kv-pos-split; a full head: rank = id)
    const PosMap pm = s.kv_pm[kv];
    const int p0l = pm_rank(pm, p0), splitl = pm_rank(pm, v.split);
    const int L = pm_rank(pm, p0 + m);
    const int k0wa = dec8_first(p0, s.win), k0w = pm_rank(pm, k0wa);
    const int chunk = dec8_chunk_pm(s, kv, p0l - k0w, p0 - k0wa);
    const int kbeg = k0w + split * chunk, kend = min(L, kbeg + chunk);
    float* wacc = work + ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 256;
    float* wml = work + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256 +
                 ((size_t)(kv * kAttnSplits + split) * kMaxGroup) * 2;
    if (kbeg >= kend) {
        for (int r = rt0 * 16 + tid; r < min(G, (rt0 + MT) * 16); r += blockDim.x) {
            wml[r * 2] = -FLT_MAX;
            wml[r * 2 + 1] = 0.f;
        }
        return;
    }
    auto row_q = [&](int r, int& qrow, int& qpos) {
        const int rr = min(r, G - 1);
        const int hl = rr / m, mi = rr % m;
        qrow = mi * s.nq + s.kv_qb[kv] + hl;
        qpos = p0 + mi;
    };
    const int d0 = warp * 64;  // this warp's dims
    // Q fragments for its 64 dims (4 chunks of 16), dims permuted like attn_dec8
    uint32_t qf[MT][4][4];
    int qpos[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        int qa, qb2;
        row_q((rt0 + mt) * 16 + g, qa, qpos[mt][0]);
        row_q((rt0 + mt) * 16 + g + 8, qb2, qpos[mt][1]);
        const bf16* ra = q + (size_t)qa * D;
        const bf16* rb = q + (size_t)qb2 * D;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int d = d0 + kk * 16 + 4 * t;
            auto cv = [](const bf16* p) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(p));
                return pack_f16(f.x, f.y);
            };
            qf[mt][kk][0] = cv(ra + d);
            qf[mt][kk][1] = cv(rb + d);
            qf[mt][kk][2] = cv(ra + d + 2);
            qf[mt][kk][3] = cv(rb + d + 2);
        }
    }
    const int ntiles = (kend - kbeg + kD8Tile - 1) / kD8Tile;
    const uint8_t* kcb = kc + head_rows(s, kv, v.max_t) * D;
    const uint8_t* vcb = vc + head_rows(s, kv, v.max_t) * D;
    const uint8_t* kcb2 = (const uint8_t*)v.kc2 + head_rows(s, kv, v.split) * D;
    const uint8_t* vcb2 = (const uint8_t*)v.vc2 + head_rows(s, kv, v.split) * D;
    auto load_tile = [&](int stage, int tile) {
        const int k0 = kbeg + tile * kD8Tile;
        uint8_t* ks = stg + (size_t)stage * 2 * kD8Tile * D;
        uint8_t* vs = ks + kD8Tile * D;
        for (int i = tid; i < kD8Tile * 16; i += kD8Warps * 32) {
            const int r = i >> 4, c = i & 15;
            const int key = k0 + r;
            const bool ok = key < kend;
            const int kk = min(key, kend - 1);
            const bool pre = kk < splitl;  // shared prefix region
            const size_t off = (size_t)(pre ? kk : kk - splitl) * D + c * 16;
            cp_async16(ks + r * D + ((c ^ (r & 7)) << 4), (pre ? kcb2 : kcb) + off, ok);
            cp_async16(vs + r * D + ((c ^ ((r >> 1) & 7)) << 4), (pre ? vcb2 : vcb) + off, ok);
        }
    };
    float o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int j = 0; j < 8; j++) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.f;
    float mx[MT][2], l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) mx[mt][0] = mx[mt][1] = -FLT_MAX, l[mt][0] = l[mt][1] = 0.f;
    const float kscale = s.ks * 256.f, vscale = s.vs * 256.f;
    const float scale = 0.0625f * 1.4426950408889634f * kscale;

#pragma unroll
    for (int sidx = 0; sidx < kD8Stages - 1; sidx++) {
        if (sidx < ntiles) load_tile(sidx, sidx);
        cp_commit();
    }
    for (int tile = 0; tile < ntiles; tile++) {
        const int nt = tile + kD8Stages - 1;
        if (nt < ntiles) load_tile(nt % kD8Stages, nt);
        cp_commit();
        cp_wait<kD8Stages - 1>();
        __syncthreads();
        const uint8_t* ks = stg + (size_t)(tile % kD8Stages) * 2 * kD8Tile * D;
        const uint8_t* vs = ks + kD8Tile * D;
        // partial scores over this warp's 64 dims (fp16-accumulate MMA: GA102 runs it at twice
        // the fp32-accumulate rate; 64-term partials are summed in fp32 below)
        float sc[MT][kD8J][4];
        uint32_t sh[MT][kD8J][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int j = 0; j < kD8J; j++) sh[mt][j][0] = sh[mt][j][1] = 0u;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int c = warp * 4 + kk;  // 16-byte chunk = 16 dims
#pragma unroll
            for (int j = 0; j < kD8J; j++) {
                const int r = j * 8 + g;
                const uint32_t w = *reinterpret_cast<const uint32_t*>(ks + r * D + ((c ^ (r & 7)) << 4) + 4 * t);
                const uint32_t b0 = e4m3x2_to_f16x2(__byte_perm(w, 0, 0x4140));
                const uint32_t b1 = e4m3x2_to_f16x2(__byte_perm(w, 0, 0x4342));
#pragma unroll
                for (int mt = 0; mt < MT; mt++) mma_hacc(sh[mt][j], qf[mt][kk], b0, b1);
            }
        }
        // exchange partial scores (fp16 MMA results): sx[warp][mt][j][lane]
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int j = 0; j < kD8J; j++)
                sx[(((size_t)warp * MT + mt) * kD8J + j) * 32 + lane] = make_uint2(sh[mt][j][0], sh[mt][j][1]);
        __syncthreads();
        // row tile mt's softmax is computed once, by warp mt (its owner: keeps mx / l), and its
        // P fragments and rescale factors are shared with the other warps through smem
        const int kt0 = kbeg + tile * kD8Tile;
        const bool tail = kt0 + kD8Tile > p0l;  // only these tiles need the absolute position
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            if (warp != mt) continue;
#pragma unroll
            for (int j = 0; j < kD8J; j++) {
                float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                for (int w2 = 0; w2 < kD8Warps; w2++) {  // fixed order, in fp32
                    const uint2 hb = sx[(((size_t)w2 * MT + mt) * kD8J + j) * 32 + lane];
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&hb.x));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&hb.y));
                    a.x += lo.x; a.y += lo.y; a.z += hi.x; a.w += hi.y;
                }
                sc[mt][j][0] = a.x; sc[mt][j][1] = a.y; sc[mt][j][2] = a.z; sc[mt][j][3] = a.w;
            }
            float tmax[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
            for (int j = 0; j < kD8J; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = kt0 + j * 8 + 2 * t + (e & 1);
                    const int qp = (e < 2) ? qpos[mt][0] : qpos[mt][1];
                    float v = sc[mt][j][e] * scale;
                    if (key >= kend || (tail && pm_unrank(pm, key) > qp)) v = -FLT_MAX;
                    sc[mt][j][e] = v;
                    tmax[e >> 1] = fmaxf(tmax[e >> 1], v);
                }
            float al[2];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
                tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
                const float nm = fmaxf(mx[mt][r], tmax[r]);
                al[r] = (mx[mt][r] == -FLT_MAX) ? 0.f : exp2f(mx[mt][r] - nm);
                mx[mt][r] = nm;
            }
            float rs[2] = {0.f, 0.f};
#pragma unroll
            for (int j = 0; j < kD8J; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float v = sc[mt][j][e];
                    const float pv = (v == -FLT_MAX) ? 0.f : exp2f(v - mx[mt][e >> 1]);
                    sc[mt][j][e] = pv;
                    rs[e >> 1] += pv;
                }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
                rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
                l[mt][r] = l[mt][r] * al[r] + rs[r];
            }
            uint32_t* dst = spa + ((size_t)mt * 32 + lane) * PW;
#pragma unroll
            for (int j = 0; j < kD8J; j++) {
                dst[j * 2] = pack_f16(sc[mt][j][0], sc[mt][j][1]);
                dst[j * 2 + 1] = pack_f16(sc[mt][j][2], sc[mt][j][3]);
            }
            *reinterpret_cast<float2*>(dst + kD8J * 2) = make_float2(al[0], al[1]);
        }
        __syncthreads();
        // A fragments of P per 16-key step: {j=2s (rows g, g+8), j=2s+1}
        uint32_t pa[MT][kD8Tile / 16][4];
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const uint32_t* src = spa + ((size_t)mt * 32 + lane) * PW;
#pragma unroll
            for (int s16 = 0; s16 < kD8Tile / 16; s16++) {
                const uint4 pk = *reinterpret_cast<const uint4*>(src + s16 * 4);
                pa[mt][s16][0] = pk.x; pa[mt][s16][1] = pk.y; pa[mt][s16][2] = pk.z; pa[mt][s16][3] = pk.w;
            }
            const float2 al = *reinterpret_cast<const float2*>(src + kD8J * 2);
            alpha[mt][0] = al.x; alpha[mt][1] = al.y;
        }
        // O += P V over this warp's 64 dims (8 n8 tiles); each V fragment serves all row tiles
#pragma unroll
        for (int nd = 0; nd < 8; nd++) {
            const int d = d0 + nd * 8 + g, c = d >> 4, cb = d & 15;
            uint32_t bv[kD8Tile / 16][2];
#pragma unroll
            for (int s16 = 0; s16 < kD8Tile / 16; s16++) {
                const int ka = s16 * 16 + 2 * t, kb2 = ka + 8;
                const int oa = ((c ^ ((ka >> 1) & 7)) << 4) + cb, ob = ((c ^ ((kb2 >> 1) & 7)) << 4) + cb;
                const uint32_t x0 = (uint32_t)vs[ka * D + oa] | ((uint32_t)vs[(ka + 1) * D + oa] << 16);
                const uint32_t x1 = (uint32_t)vs[kb2 * D + ob] | ((uint32_t)vs[(kb2 + 1) * D + ob] << 16);
                bv[s16][0] = e4m3x2_to_f16x2(x0);
                bv[s16][1] = e4m3x2_to_f16x2(x1);
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {  // o = o * alpha + (P V of this tile, fp16 acc)
                uint32_t c2[2] = {0u, 0u};
#pragma unroll
                for (int s16 = 0; s16 < kD8Tile / 16; s16++) mma_hacc(c2, pa[mt][s16], bv[s16][0], bv[s16][1]);
                const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&c2[0]));
                const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&c2[1]));
                o[mt][nd][0] = o[mt][nd][0] * alpha[mt][0] + lo.x;
                o[mt][nd][1] = o[mt][nd][1] * alpha[mt][0] + lo.y;
                o[mt][nd][2] = o[mt][nd][2] * alpha[mt][1] + hi.x;
                o[mt][nd][3] = o[mt][nd][3] * alpha[mt][1] + hi.y;
            }
        }
        __syncthreads();  // sx and this stage are reused
    }
    cp_wait<0>();
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = (rt0 + mt) * 16 + g + r * 8;
            if (row >= G) continue;
#pragma unroll
            for (int nd = 0; nd < 8; nd++) {
                const int d = d0 + nd * 8 + 2 * t;
                *reinterpret_cast<float2*>(wacc + (size_t)row * 256 + d) =
                    make_float2(o[mt][nd][r * 2] * vscale, o[mt][nd][r * 2 + 1] * vscale);
            }
            if (warp == mt && t == 0) {  // the row tile's softmax owner
                wml[row * 2] = mx[mt][r] == -FLT_MAX ? -FLT_MAX : mx[mt][r] * 0.6931471805599453f;
                wml[row * 2 + 1] = l[mt][r];
            }
        }
}

template <int MT>
__global__ void __launch_bounds__(kD8Warps * 32, 2) attn_dec8d_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    AttnShape s, const int* __restrict__ pos, int m, float* __restrict__ work, int rs = 0) {
    attn_dec8d_dev<MT>(q, kc, vc, s, seq_view(s), pos, m, work, blockIdx.y % kAttnSplits, (blockIdx.y / kAttnSplits) * MT, rs);
}

template <int MT>
__global__ void __launch_bounds__(kD8Warps * 32, 2) attn_dec8d_seg_kernel(const AttnSegs gs, const AttnShape s, int rs = 0) {
    const AttnSeg& c = gs.c[blockIdx.z];
    attn_dec8d_dev<MT>(c.q, (const uint8_t*)c.kc, (const uint8_t*)c.vc, s, seq_view(c), c.pos, c.m, c.work,
                       blockIdx.y % kAttnSplits, (blockIdx.y / kAttnSplits) * MT, rs);
}

template <int RT, int KG>
__global__ void __launch_bounds__(RT * KG * 32) attn_dec8_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    AttnShape s, const int* __restrict__ pos, int m, float* __restrict__ work) {
    attn_dec8_dev<RT, KG>(q, kc, vc, s, seq_view(s), pos, m, work);
}

template <int RT, int KG>
__global__ void __launch_bounds__(RT * KG * 32) attn_dec8_seg_kernel(const AttnSegs g, const AttnShape s) {
    const AttnSeg& c = g.c[blockIdx.z];
    attn_dec8_dev<RT, KG>(c.q, (const uint8_t*)c.kc, (const uint8_t*)c.vc, s, seq_view(c), c.pos, c.m, c.work);
}

template <int RT, int KG>
void launch_dec8_seg(const AttnSegs& g, const AttnShape& s, cudaStream_t st) {
    attn_dec8_seg_kernel<RT, KG><<<dim3(s.nkv, kAttnSplits, g.n), RT * KG * 32, dec8_smem<RT, KG>(), st>>>(g, s);
}

template <int RT, int KG>
void launch_dec8(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const int* pos,
                 int m, float* work, cudaStream_t st) {
    attn_dec8_kernel<RT, KG><<<dim3(s.nkv, kAttnSplits), RT * KG * 32, dec8_smem<RT, KG>(), st>>>(
        q, (const uint8_t*)kc, (const uint8_t*)vc, s, pos, m, work);
}
template <int RT, int KG>
void set_dec8() {
    CUDA_CHECK(cudaFuncSetAttribute(attn_dec8_kernel<RT, KG>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)dec8_smem<RT, KG>()));
    CUDA_CHECK(cudaFuncSetAttribute(attn_dec8_seg_kernel<RT, KG>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)dec8_smem<RT, KG>()));
}
constexpr int kDecKG = 2;

template <int NW, bool F8, bool DEC>
constexpr size_t smem_for() { return smem_bytes<NW, F8, bkv_for<F8, DEC>(), !DEC>(); }

template <int NW, bool F8, bool DEC>
void set_attr() {
    CUDA_CHECK(cudaFuncSetAttribute(attn_tc_kernel<NW, F8, DEC>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_for<NW, F8, DEC>()));
}

template <int NW, bool F8, bool DEC>
void launch(dim3 grid, const bf16* q, const void* kc, const void* vc, const AttnShape& s,
            const float* proj, int ld, const int* pos, int m, bf16* out, float* work,
            cudaStream_t st) {
    attn_tc_kernel<NW, F8, DEC><<<grid, threads_for<NW>(), smem_for<NW, F8, DEC>(), st>>>(
        q, kc, vc, s, proj, ld, pos, m, out, work);
}

// Per-key int8 scales of the fp8 K cache for the QK8 prefill kernel, once per call instead of once per
// block: warp per local key row l < rank(pos[0] + m) of each local kv head, max |k| over the 256 dims
// (raw e4m3 bytes, as in attn_pf8_kernel's kscales), out[kv * ld + l] = (127 / max as half2, 1 / that).
// K smoothing: per-channel sums of the call's local key rows (fp16 values of the e4m3 cache * 2^-8) into
// acc[kv][256] (zeroed before), block = 256 channels, keys strided over the grid
__global__ void pattn_kmean_kernel(const uint8_t* __restrict__ kc, AttnShape s, const int* __restrict__ pos, int m,
                                   float* __restrict__ acc) {
    const int kv = blockIdx.y, d = threadIdx.x;
    const PosMap pm = s.kv_pm[kv];
    const int rows = pm_rank(pm, pos[0] + m), splitl = pm_rank(pm, s.split);
    const uint8_t* kb = kc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* kb2 = (const uint8_t*)s.kc2 + head_rows(s, kv, s.split) * D;
    float a = 0.f;
    for (int l = blockIdx.x; l < rows; l += gridDim.x) {
        const uint32_t b = (l < splitl ? kb2 + (size_t)l * D : kb + (size_t)(l - splitl) * D)[d];
        const unsigned short hb = (unsigned short)(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8));
        a += __half2float(__ushort_as_half(hb));
    }
    atomicAdd(acc + kv * D + d, a);
}

__global__ void pattn_kscale_kernel(const uint8_t* __restrict__ kc, AttnShape s, const int* __restrict__ pos, int m,
                                    uint2* __restrict__ out, int ld) {
    const int kv = blockIdx.y, lane = threadIdx.x & 31;
    const PosMap pm = s.kv_pm[kv];
    const int rows = pm_rank(pm, pos[0] + m), splitl = pm_rank(pm, s.split);
    const uint8_t* kb = kc + head_rows(s, kv, s.max_t) * D;
    const uint8_t* kb2 = (const uint8_t*)s.kc2 + head_rows(s, kv, s.split) * D;
    const float kinv_rows = 1.f / (float)max(1, rows);
    for (int l = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); l < rows; l += gridDim.x * (blockDim.x / 32)) {
        const uint8_t* kr = l < splitl ? kb2 + (size_t)l * D : kb + (size_t)(l - splitl) * D;
        const uint2 w = __ldg(reinterpret_cast<const uint2*>(kr) + lane);
        int km;
        if (s.kmu) {  // max |k - mu| with mu as the attention kernel computes it (fp16)
            uint32_t e0, o0, e1, o1;
            f8x4_to_f16(w.x, e0, o0);
            f8x4_to_f16(w.y, e1, o1);
            uint32_t mu[8];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int d = lane * 8 + 4 * j;
                auto mf = [&](int dd) { return __float2half_rn(s.kmu[kv * D + dd] * kinv_rows); };
                const __half2 a2 = __halves2half2(mf(d), mf(d + 2)), b2 = __halves2half2(mf(d + 1), mf(d + 3));
                mu[2 * j] = *reinterpret_cast<const uint32_t*>(&a2);
                mu[2 * j + 1] = *reinterpret_cast<const uint32_t*>(&b2);
            }
            const uint32_t hm = h2_max(h2_max(h2_sub(e0, mu[0]) & 0x7FFF7FFFu, h2_sub(o0, mu[1]) & 0x7FFF7FFFu),
                                       h2_max(h2_sub(e1, mu[2]) & 0x7FFF7FFFu, h2_sub(o1, mu[3]) & 0x7FFF7FFFu));
            km = max((int)(hm & 0xFFFF), (int)(hm >> 16));  // fp16 magnitudes order as integers
        } else {
            const uint32_t hm = h2_max(h2_max(w.x & 0x007F007Fu, (w.x >> 8) & 0x007F007Fu),
                                       h2_max(w.y & 0x007F007Fu, (w.y >> 8) & 0x007F007Fu));
            km = max((int)(hm & 0xFFFF), (int)(hm >> 16)) << 7;  // the byte as fp16 * 2^-8
        }
#pragma unroll
        for (int o2 = 1; o2 < 32; o2 <<= 1) km = max(km, __shfl_xor_sync(0xffffffffu, km, o2));
        if (lane == 0) {
            const float kmax = __half2float(__ushort_as_half((unsigned short)km));
            const __half ih = __float2half_rn(kmax > 0.f ? fminf(127.f / kmax, 65504.f) : 0.f);
            const __half2 ih2 = __halves2half2(ih, ih);
            const float fi = __half2float(ih);
            out[(size_t)kv * ld + l] = make_uint2(*reinterpret_cast<const uint32_t*>(&ih2), __float_as_uint(fi > 0.f ? 1.f / fi : 0.f));
        }
    }
}

// ---------------------------------------------------------------- windowed precomputed K / V (QE_PATTN_KVPRE)
// The call's local key rows are processed in windows of s.kvw rows: pattn_kvpre_kernel converts the window
// once (K -> int8 with the per-key scales of pattn_kscale_kernel, V -> fp16 in attn_pf8_kernel's dim
// permutation, both bit for bit what the in-kernel widening produces) into scratch, then attn_pkv_kernel
// (= attn_pf8_kernel<QK8> without the widening: 64-key tiles copied by cp.async straight into a double
// buffer, one barrier per tile) runs on the window. Each block keeps its split slot of the prefill
// partials and merges the windows into it (window 0 writes), so the stats exchange / combine are unchanged.
constexpr int kPKBuf = 3072;                                // uint4 per buffer: K int8 [64][16] | V fp16 [64][32]
constexpr int kPKSK = 2 * kPKBuf;                           // scales [2][64] float2
constexpr size_t kPKSmem = (size_t)(kPKSK + 64) * 16;       // 99328 B

__global__ void pattn_kvpre_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, AttnShape s,
                                   const int* __restrict__ pos, int m, int w0, uint8_t* __restrict__ kq,
                                   uint4* __restrict__ vh) {
    const int kv = blockIdx.y, lane = threadIdx.x & 31, W = s.kvw;
    const PosMap pm = s.kv_pm[kv];
    const int rows = min(pm_rank(pm, pos[0] + m), w0 + W), splitl = pm_rank(pm, s.split);
    const int r = w0 + blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const size_t ho = head_rows(s, kv, s.max_t) * D, ho2 = head_rows(s, kv, s.split) * D;
    const uint8_t* kr = r < splitl ? (const uint8_t*)s.kc2 + ho2 + (size_t)r * D : kc + ho + (size_t)(r - splitl) * D;
    const uint8_t* vr = r < splitl ? (const uint8_t*)s.vc2 + ho2 + (size_t)r * D : vc + ho + (size_t)(r - splitl) * D;
    const uint2 kw = __ldg(reinterpret_cast<const uint2*>(kr) + lane);
    const uint2 vw = __ldg(reinterpret_cast<const uint2*>(vr) + lane);
    const uint32_t inv2 = s.ksc[(size_t)kv * s.ksc_ld + r].x;
    uint32_t e0, o0, e1, o1;
    f8x4_to_f16(kw.x, e0, o0);
    f8x4_to_f16(kw.y, e1, o1);
    reinterpret_cast<uint2*>(kq + ((size_t)kv * W + (r - w0)) * D)[lane] =
        make_uint2(__byte_perm(h2_q8(e0, inv2), h2_q8(o0, inv2), 0x6240), __byte_perm(h2_q8(e1, inv2), h2_q8(o1, inv2), 0x6240));
    f8x4_to_f16(vw.x, e0, o0);
    f8x4_to_f16(vw.y, e1, o1);
    vh[((size_t)kv * W + (r - w0)) * 32 + lane] = make_uint4(e0, o0, e1, o1);
}

__global__ void __launch_bounds__(256, 1) attn_pkv_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kq, const uint4* __restrict__ vh, AttnShape s,
    const int* __restrict__ pos, int m, float* __restrict__ work, int w0, int wi) {
    extern __shared__ __align__(128) uint4 sm[];
    constexpr int kRows = 128, kKeys = 64, NT = 256, NJ = 8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0], W = s.kvw;
    const int QT = (m + kRows - 1) / kRows, S = s.pf_splits;
    const int qt = blockIdx.x % QT, sp = (blockIdx.x / QT) % S, hh = blockIdx.x / (QT * S);
    int h = 0;
    {
        int kvo[kMaxLocalKv];
        for (int j = 0; j < s.nkv; j++) kvo[j] = j;
        for (int a = 1; a < s.nkv; a++)
            for (int b = a; b > 0; b--) {
                const PosMap& x = s.kv_pm[kvo[b]];
                const PosMap& y = s.kv_pm[kvo[b - 1]];
                if ((long long)x.A * y.P > (long long)y.A * x.P) { const int tt = kvo[b]; kvo[b] = kvo[b - 1]; kvo[b - 1] = tt; }
                else break;
            }
        int rem = hh;
        for (int j = 0; j < s.nkv; j++) {
            const int kj = kvo[j], n = s.kv_qe[kj] - s.kv_qb[kj];
            if (rem < n) { h = s.kv_qb[kj] + rem; break; }
            rem -= n;
        }
    }
    const int q0 = qt * kRows, kv = s.q2kv[h];
    const int G = min(kRows, m - q0);
    const PosMap pm = s.kv_pm[kv];
    const int kendb = pm_rank(pm, p0 + min(q0 + kRows, m));
    const int wend = min(w0 + W, kendb);
    int chunk = (W + S - 1) / S;
    chunk = (chunk + kKeys - 1) / kKeys * kKeys;
    const int kbeg = w0 + sp * chunk, kend = min(wend, kbeg + chunk);
    const int split = sp;
    auto pacc = [&](int row) { return work + (((size_t)split * m + q0 + row) * s.nq + h) * 256; };
    auto pml = [&](int row) {
        return work + (size_t)S * m * s.nq * 256 + (((size_t)split * m + q0 + row) * s.nq + h) * 2;
    };
    if (kbeg >= kend) {
        if (wi == 0)
            for (int r = tid; r < G; r += NT) {
                pml(r)[0] = -FLT_MAX;
                pml(r)[1] = 0.f;
            }
        return;
    }
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    __shared__ float s_sq[kRows];
    {  // Q -> int8 per row (as attn_pf8_kernel<true>)
        const float sc = 0.0625f * 1.4426950408889634f * s.ks * 256.f;
        for (int i = tid; i < kRows * 16; i += NT) {
            const int r = i >> 4, ch = i & 15;
            const int qi = q0 + min(r, G - 1);
            const uint4* src = reinterpret_cast<const uint4*>(q + ((size_t)qi * s.nq + h) * D) + 2 * ch;
            const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float am = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[j]));
                am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o2));
            const float qa = am * fabsf(sc);
            const float sq = qa > 0.f ? qa / 127.f : 1.f;
            const float qinv = sc / sq;
            if (ch == 0) s_sq[r] = sq;
            uint32_t o4[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f0 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j]));
                const float2 f1 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j + 1]));
                const int b0 = __float2int_rn(f0.x * qinv), b1 = __float2int_rn(f0.y * qinv);
                const int b2 = __float2int_rn(f1.x * qinv), b3 = __float2int_rn(f1.y * qinv);
                o4[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((uint32_t)(b3 & 0xFF) << 24);
            }
            sm[r * 16 + (ch ^ (r & 7))] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
    }
    __syncthreads();
    uint32_t qf[D / 32][4];
#pragma unroll
    for (int kk = 0; kk < D / 32; kk++) {
        const int r = warp * 16 + (lane & 15), ch = kk * 2 + (lane >> 4);
        ldsm_x4(qf[kk], &sm[r * 16 + (ch ^ (r & 7))]);
    }
    __syncthreads();

    const uint8_t* kqb = kq + (size_t)kv * W * D;
    const uint4* vhb = vh + (size_t)kv * W * 32;
    const uint2* kscr = s.ksc + (size_t)kv * s.ksc_ld;
    const int cr = tid >> 4, cc = tid & 15, vr = tid >> 5, vc = tid & 31;
    const int wsw8 = cr * 16 + (cc ^ (cr & 7));
    const int ntiles = (kend - kbeg + kKeys - 1) / kKeys;
    auto load = [&](int tile, int buf) {
        const int k0 = kbeg + tile * kKeys;
        uint4* kb = sm + buf * kPKBuf;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int key = k0 + cr + 16 * u;
            cp_async16m(kb + wsw8 + 256 * u, kqb + (size_t)(min(key, kend - 1) - w0) * D + cc * 16, key < kend);
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const int r = vr + 8 * u, key = k0 + r;
            cp_async16m(kb + 1024 + sw(r, vc), vhb + (size_t)(min(key, kend - 1) - w0) * 32 + vc, key < kend);
        }
        if (tid < kKeys) {
            const float skq = __uint_as_float(kscr[min(k0 + tid, kend - 1)].y);
            reinterpret_cast<float2*>(sm + kPKSK)[buf * kKeys + tid] = make_float2(skq, -(12582912.f * skq));
        }
    };
    const int mi = lane >> 3, L7 = lane & 7;
    unsigned kq4[4], vq4[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        kq4[i] = sbase + 16u * (((mi >> 1) * 8 + L7) * 16 + ((2 * i + (mi & 1)) ^ L7));
        vq4[i] = sbase + 16u * (1024 + ((mi & 1) * 8 + L7) * 32 + ((2 * i + (mi >> 1)) ^ L7));
    }
    load(0, 0);
    cp_commit();

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int r0 = warp * 16 + g;
    const int qpos0 = p0 + q0 + min(r0, G - 1), qpos1 = p0 + q0 + min(r0 + 8, G - 1);
    const int rl0 = pm_rank(pm, p0 + q0);
    const float vscale = s.vs * 256.f;
    const float sq0 = s_sq[r0], sq1 = s_sq[r0 + 8];
    constexpr uint32_t kOnes = 0x3C003C00u;

    for (int tile = 0; tile < ntiles; tile++) {
        const int k0 = kbeg + tile * kKeys, b = tile & 1;
        cp_wait<0>();
        __syncthreads();  // tile ready everywhere; tile - 1's buffer free
        if (tile + 1 < ntiles) load(tile + 1, b ^ 1);
        cp_commit();
        const unsigned bo = (unsigned)(b * kPKBuf * 16);
        uint32_t si[NJ][4];
#pragma unroll
        for (int j = 0; j < NJ; j++) si[j][0] = si[j][1] = si[j][2] = si[j][3] = 0x4B400000u;
#pragma unroll
        for (int kk = 0; kk < D / 32; kk++)
#pragma unroll
            for (int jp = 0; jp < 4; jp++) {
                uint32_t bb[4];
                ldsm_x4a(bb, kq4[kk & 3] + bo + jp * 4096 + (kk >> 2) * 128);
                imma(si[jp * 2], qf[kk], bb[0], bb[1]);
                imma(si[jp * 2 + 1], qf[kk], bb[2], bb[3]);
            }
        float sf[NJ][4];
        const float4* skp = reinterpret_cast<const float4*>(sm + kPKSK) + b * (kKeys / 2);
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const float4 k4 = skp[j * 4 + t];
            sf[j][0] = fmaf(__uint_as_float(si[j][0]), k4.x, k4.y);
            sf[j][1] = fmaf(__uint_as_float(si[j][1]), k4.z, k4.w);
            sf[j][2] = fmaf(__uint_as_float(si[j][2]), k4.x, k4.y);
            sf[j][3] = fmaf(__uint_as_float(si[j][3]), k4.z, k4.w);
        }
        if (k0 + kKeys > min(kend, rl0)) {
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = k0 + j * 8 + 2 * t + (e & 1);
                    if (key >= kend || pm_unrank(pm, key) > ((e < 2) ? qpos0 : qpos1)) sf[j][e] = -INFINITY;
                }
        }
        uint32_t sc[NJ][2];
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float tmax = -INFINITY;
#pragma unroll
            for (int j = 0; j < NJ; j++) tmax = fmaxf(tmax, fmaxf(sf[j][2 * r], sf[j][2 * r + 1]));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float qs = r ? sq1 : sq0;
            const float nm = fmaxf(mx[r], tmax * qs);
            alpha[r] = mx[r] == -INFINITY ? 0.f : exp2f(mx[r] - nm);
            mx[r] = nm;
            const float mm = nm == -INFINITY ? 0.f : nm;
#pragma unroll
            for (int j = 0; j < NJ; j++)
                sc[j][r] = pack_f16(exp2f(fmaf(sf[j][2 * r], qs, -mm)), exp2f(fmaf(sf[j][2 * r + 1], qs, -mm)));
        }
        {
            uint32_t cl[2] = {0u, 0u};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                mma_hacc(cl, a, kOnes, kOnes);
            }
            l[0] = l[0] * alpha[0] + __low2float(*reinterpret_cast<__half2*>(&cl[0]));
            l[1] = l[1] * alpha[1] + __low2float(*reinterpret_cast<__half2*>(&cl[1]));
        }
#pragma unroll
        for (int jp = 0; jp < 16; jp++) {
            uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u}, d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                uint32_t bb[4];
                ldsm_x4ta(bb, vq4[jp & 3] + bo + ks * 8192 + (jp >> 2) * 128);
                if (ks >= 2) {
                    mma_hacc(d0, a, bb[0], bb[1]);
                    mma_hacc(d1, a, bb[2], bb[3]);
                } else {
                    mma_hacc(c0, a, bb[0], bb[1]);
                    mma_hacc(c1, a, bb[2], bb[3]);
                }
            }
            const float2 a0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0])), b0 = __half22float2(*reinterpret_cast<__half2*>(&d0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1])), b1 = __half22float2(*reinterpret_cast<__half2*>(&d0[1]));
            const float2 a2 = __half22float2(*reinterpret_cast<__half2*>(&c1[0])), b2 = __half22float2(*reinterpret_cast<__half2*>(&d1[0]));
            const float2 a3 = __half22float2(*reinterpret_cast<__half2*>(&c1[1])), b3 = __half22float2(*reinterpret_cast<__half2*>(&d1[1]));
            float* pa = o[jp * 2];
            float* pb = o[jp * 2 + 1];
            pa[0] = pa[0] * alpha[0] + (a0.x + b0.x); pa[1] = pa[1] * alpha[0] + (a0.y + b0.y);
            pa[2] = pa[2] * alpha[1] + (a1.x + b1.x); pa[3] = pa[3] * alpha[1] + (a1.y + b1.y);
            pb[0] = pb[0] * alpha[0] + (a2.x + b2.x); pb[1] = pb[1] * alpha[0] + (a2.y + b2.y);
            pb[2] = pb[2] * alpha[1] + (a3.x + b3.x); pb[3] = pb[3] * alpha[1] + (a3.y + b3.y);
        }
    }
    cp_wait<0>();

    // ---- partial of this window: window 0 writes, later windows merge into the slot
    const bool odd = t & 1;
    const int dbase = 4 * (t >> 1) + (odd ? 2 : 0);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = r0 + 8 * r;
        float v[32][2];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const float x0 = o[j][2 * r], x1 = o[j][2 * r + 1];
            const float rcv = __shfl_xor_sync(0xffffffffu, odd ? x0 : x1, 1);
            v[j][0] = odd ? rcv : x0;
            v[j][1] = odd ? x1 : rcv;
        }
        const unsigned live = __ballot_sync(0xffffffffu, row < G);
        if (row >= G) continue;
        const float mn = mx[r] == -INFINITY ? -FLT_MAX : mx[r] * 0.6931471805599453f, ln = l[r];
        float fo = 0.f, fn = 1.f, mo = -FLT_MAX, lo = 0.f;
        if (wi > 0) {
            mo = pml(row)[0];
            lo = pml(row)[1];
            const float M = fmaxf(mo, mn);
            fo = (mo == -FLT_MAX || lo == 0.f) ? 0.f : __expf(mo - M);
            fn = (mn == -FLT_MAX || ln == 0.f) ? 0.f : __expf(mn - M);
        }
        float* pr = pacc(row);
#pragma unroll
        for (int j = 0; j < 32; j++) {
            float2 nv = make_float2(v[j][0] * vscale * fn, v[j][1] * vscale * fn);
            if (wi > 0 && fo != 0.f) {
                const float2 ov = *reinterpret_cast<const float2*>(pr + j * 8 + dbase);
                nv.x += ov.x * fo;
                nv.y += ov.y * fo;
            }
            *reinterpret_cast<float2*>(pr + j * 8 + dbase) = nv;
        }
        __syncwarp(live);
        if (t == 0) {
            if (wi == 0) {
                pml(row)[0] = mn;
                pml(row)[1] = ln;
            } else {
                const float M = fmaxf(mo, mn);
                pml(row)[0] = (fo == 0.f && fn == 0.f) ? -FLT_MAX : M;
                pml(row)[1] = lo * fo + ln * fn;
            }
        }
    }
}

// ---------------------------------------------------------------- persistent windowed prefill attention
// One launch of gridDim.x = #SM blocks (1 per SM) pulls items from a work queue (atomic counter) in the order
//   QPREP(qt, h) x QT*nq | CONV(0) | ATTN(0) | CONV(1) | ATTN(1) | ... | CONV(nwin-1) | ATTN(nwin-1)
// QPREP: Q -> int8 MMA fragments + row scales into scratch (once per call instead of once per item).
// CONV(w): 128 key rows of one kv head of window w -> K int8 (per-key scale computed here, as
//   pattn_kscale_kernel) + V fp16 into window buffer w & 1. Waits until all ATTN(w - 2) items released
//   that buffer.
// ATTN(w): attn_pkv_kernel's work for (q tile, head, split) on window w; waits for CONV(w). Its partial is
//   merged into the split slot in window order (slot flag = windows merged), so the result is deterministic.
// Every wait is on items earlier in the queue, which were taken by running blocks: no deadlock even if not
// all blocks are resident. The conversion of window w + 1 runs on the SMs that free up in the tail of
// window w's attention; no per-window launch tails.
constexpr int kPkpConvRows = 128;
constexpr int kPkpMaxWin = 64;
constexpr int kPkpCtrNext = 0, kPkpCtrQ = 1, kPkpCtrConv = 2, kPkpCtrAttn = 2 + kPkpMaxWin, kPkpCtrSlot = 2 + 2 * kPkpMaxWin;

__device__ __forceinline__ int pkp_ld_acq(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void pkp_st_rel(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ volatile int* g_pkp_diag;  // QE_PKP_DIAG: host-mapped per-block wait state, dumped by a host watchdog
__device__ __forceinline__ void pkp_wait_ge(const int* p, int target, int tag = 0) {
    long long spins = 0;
    unsigned ws = 0;
    while (pkp_ld_acq(p) < target) {
        __nanosleep(64);
        QE_WD_POLL(ws, kWdPkp, -1, (unsigned)target, (unsigned)pkp_ld_acq(p));
        if (++spins == (1ll << 20) && g_pkp_diag) {
            volatile int* d = g_pkp_diag + blockIdx.x * 8;
            d[0] = tag; d[1] = target; d[2] = pkp_ld_acq(p); d[3] = (int)((size_t)p & 0xFFFFF); d[4] = 1;
            __threadfence_system();
        }
    }
    if (spins >= (1ll << 20) && g_pkp_diag) g_pkp_diag[blockIdx.x * 8 + 4] = 0;
}

#ifndef QE_PKP_L32
#define QE_PKP_L32 1
#endif
__constant__ int s_pkp_cpos;  // QE_PKP_CPOS: % of ATTN(w) items queued before CONV(w + 1)
__device__ unsigned long long* g_pkp_trace;  // bench only (qe_pkp_trace): per item [it, sm, t0, t1, t2, t3, t4, kind]
__device__ __forceinline__ unsigned long long pkp_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned pkp_smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %smid;" : "=r"(r));
    return r;
}

__global__ void __launch_bounds__(256, 1) attn_pkp_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, AttnShape s,
    PattnScratch ps, const int* __restrict__ pos, int m, float* __restrict__ work, int nwin) {
    extern __shared__ __align__(128) uint4 sm[];
    constexpr int kRows = 128, kKeys = 64, NT = 256, NJ = 8;
    __shared__ float s_sq[kRows];
    __shared__ int s_item;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int p0 = pos[0], W = ps.w;
    const int QT = (m + kRows - 1) / kRows, S = s.pf_splits;
    const int nQ = ps.qf ? QT * s.nq : 0, nC = s.nkv * (W / kPkpConvRows), nA = QT * s.nq * S;
    // CONV(w + 1) sits after the first cpos of ATTN(w)'s items, so it runs beside window w's attention
    const int cpos = min(nA, max(0, (int)((long long)nA * s_pkp_cpos / 100)));
    const int total = nQ + nC + (nwin - 1) * (nA + nC) + nA;
    int* ctr = ps.ctr;
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);

    // Q of (tile qt, head h) -> int8 fragments qf (as attn_pf8_kernel<true>), row scales in s_sq
    auto quant_q = [&](int qt, int h, uint32_t (&qf)[D / 32][4]) {
        const int q0 = qt * kRows, G = min(kRows, m - q0);
        const float sc = 0.0625f * 1.4426950408889634f * s.ks * 256.f;
        for (int i = tid; i < kRows * 16; i += NT) {
            const int r = i >> 4, ch = i & 15;
            const int qi = q0 + min(r, G - 1);
            const uint4* src = reinterpret_cast<const uint4*>(q + ((size_t)qi * s.nq + h) * D) + 2 * ch;
            const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float am = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[j]));
                am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o2));
            const float qa = am * fabsf(sc);
            const float sq = qa > 0.f ? qa / 127.f : 1.f;
            const float qinv = sc / sq;
            if (ch == 0) s_sq[r] = sq;
            uint32_t o4[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f0 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j]));
                const float2 f1 = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[2 * j + 1]));
                const int b0 = __float2int_rn(f0.x * qinv), b1 = __float2int_rn(f0.y * qinv);
                const int b2 = __float2int_rn(f1.x * qinv), b3 = __float2int_rn(f1.y * qinv);
                o4[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((uint32_t)(b3 & 0xFF) << 24);
            }
            sm[r * 16 + (ch ^ (r & 7))] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < D / 32; kk++) {
            const int r = warp * 16 + (lane & 15), ch = kk * 2 + (lane >> 4);
            ldsm_x4(qf[kk], &sm[r * 16 + (ch ^ (r & 7))]);
        }
        __syncthreads();
    };

    if (tid == 0) s_item = atomicAdd(ctr + kPkpCtrNext, 1);
    if (g_pkp_diag && tid == 0) atomicAdd((int*)g_pkp_diag + 2000, 1), g_pkp_diag[2002 + blockIdx.x] = 1;
    if (g_pkp_diag && tid == 0 && blockIdx.x == 0)
        g_pkp_diag[1990] = m, g_pkp_diag[1991] = p0, g_pkp_diag[1992] = nwin, g_pkp_diag[1993] = S, g_pkp_diag[1994] = W,
        g_pkp_diag[1995] = total, g_pkp_diag[1996] = nQ, g_pkp_diag[1997] = nC, g_pkp_diag[1998] = nA;
    // (thread 0) deferred slot release: the flag store of the last ATTN item waits for its partial's stores to
    // drain (fence) - done later (in the next item, before any wait that could depend on it) so the drain
    // overlaps work
    int* rel_p = nullptr;
    int rel_v = 0;
    auto release = [&]() {
        if (rel_p) {
            __threadfence();
            pkp_st_rel(rel_p, rel_v);
            rel_p = nullptr;
        }
    };
    int conv_ok = -1;     // (thread 0) window whose CONV items are known complete
    bool q_ok = nQ == 0;  // (thread 0) QPREP items known complete
    for (;;) {
        __syncthreads();
        const int it = s_item;
        __syncthreads();
        if (it >= total) {
            if (tid == 0) release();
            if (g_pkp_diag && tid == 0) atomicAdd((int*)g_pkp_diag + 2001, 1), g_pkp_diag[2002 + blockIdx.x] = 0;
            break;
        }
        if (g_pkp_diag && tid == 0) g_pkp_diag[3000 + blockIdx.x] = it;
#ifndef QE_PKP_PREGRAB
#define QE_PKP_PREGRAB 0
#endif
        int nxt = 0;  // (thread 0) the next item (QE_PKP_PREGRAB: taken now, the atomic's latency under this item)
        if (QE_PKP_PREGRAB && tid == 0) nxt = atomicAdd(ctr + kPkpCtrNext, 1);
        unsigned long long* tr = (g_pkp_trace && tid == 0 && it < 65536) ? g_pkp_trace + (size_t)it * 16 : nullptr;
        if (tr) { for (int z = 0; z < 16; z++) tr[z] = 0; tr[0] = it, tr[1] = pkp_smid(), tr[2] = pkp_now(); }
        if (it < nQ) {  // ---- QPREP
            const int qt = it % QT, h = it / QT;
            uint32_t qf[D / 32][4];
            quant_q(qt, h, qf);
            uint4* dst = ps.qf + ((size_t)(qt * s.nq + h) * 8 + warp) * (D / 32) * 32;
#pragma unroll
            for (int kk = 0; kk < D / 32; kk++) dst[kk * 32 + lane] = make_uint4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3]);
            if (tid < kRows) ps.qsq[(size_t)(qt * s.nq + h) * kRows + tid] = s_sq[tid];
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicAdd(ctr + kPkpCtrQ, 1);
            }
            if (tr) tr[6] = pkp_now(), tr[7] = 1;
            if (tid == 0) {
                release();
                s_item = QE_PKP_PREGRAB ? nxt : atomicAdd(ctr + kPkpCtrNext, 1);
            }
            continue;
        }
        int wi, j;  // j < nC: CONV(wi, j), else ATTN(wi, j - nC)
        {
            const int rr = it - nQ;
            if (rr < nC) {
                wi = 0, j = rr;
            } else {
                const int r2 = rr - nC, wa = r2 / (nA + nC), ja = r2 % (nA + nC);
                if (wa >= nwin - 1) wi = nwin - 1, j = nC + (r2 - (nwin - 1) * (nA + nC));
                else if (ja < cpos) wi = wa, j = nC + ja;
                else if (ja < cpos + nC) wi = wa + 1, j = ja - cpos;
                else wi = wa, j = ja;
            }
        }
        const int b = wi & 1, w0 = wi * W;
        if (j < nC) {  // ---- CONV(wi): 128 rows of kv head j % nkv
            const int kv = j % s.nkv, rb = w0 + (j / s.nkv) * kPkpConvRows;
            const PosMap pm = s.kv_pm[kv];
            const int rows = min(pm_rank(pm, p0 + m), w0 + W), splitl = pm_rank(pm, s.split);
            if (rb < rows) {
                if (wi >= 2) {
                    if (tid == 0) pkp_wait_ge(ctr + kPkpCtrAttn + wi - 2, nA, 1000 + wi);
                    if (tr) tr[3] = pkp_now();
                    __syncthreads();
                }
                const size_t ho = head_rows(s, kv, s.max_t) * D, ho2 = head_rows(s, kv, s.split) * D;
                uint8_t* kq = ps.kq[b][kv];
                uint4* vh = ps.vh[b][kv];
                uint2* ksc = s.ksc + (size_t)kv * s.ksc_ld;
                constexpr int RW = kPkpConvRows / 8;  // rows per warp, all loads in flight at once
                uint2 kwa[RW], vwa[RW];
#pragma unroll
                for (int i = 0; i < RW; i++) {
                    const int r = min(rb + warp * RW + i, rows - 1);
                    const uint8_t* kr = r < splitl ? (const uint8_t*)s.kc2 + ho2 + (size_t)r * D : kc + ho + (size_t)(r - splitl) * D;
                    const uint8_t* vr = r < splitl ? (const uint8_t*)s.vc2 + ho2 + (size_t)r * D : vc + ho + (size_t)(r - splitl) * D;
                    kwa[i] = __ldg(reinterpret_cast<const uint2*>(kr) + lane);
                    vwa[i] = __ldg(reinterpret_cast<const uint2*>(vr) + lane);
                }
#pragma unroll
                for (int i = 0; i < RW; i++) {
                    const int r = rb + warp * RW + i;
                    if (r >= rows) break;
                    const uint2 kw = kwa[i], vw = vwa[i];
                    // per-key scale exactly as pattn_kscale_kernel
                    const uint32_t hm = h2_max(h2_max(kw.x & 0x007F007Fu, (kw.x >> 8) & 0x007F007Fu),
                                               h2_max(kw.y & 0x007F007Fu, (kw.y >> 8) & 0x007F007Fu));
                    int km = max((int)(hm & 0xFFFF), (int)(hm >> 16)) << 7;
#pragma unroll
                    for (int o2 = 1; o2 < 32; o2 <<= 1) km = max(km, __shfl_xor_sync(0xffffffffu, km, o2));
                    const float kmax = __half2float(__ushort_as_half((unsigned short)km));
                    const __half ih = __float2half_rn(kmax > 0.f ? fminf(127.f / kmax, 65504.f) : 0.f);
                    const __half2 ih2 = __halves2half2(ih, ih);
                    const uint32_t inv2 = *reinterpret_cast<const uint32_t*>(&ih2);
                    if (lane == 0) {
                        const float fi = __half2float(ih);
                        ksc[r] = make_uint2(inv2, __float_as_uint(fi > 0.f ? 1.f / fi : 0.f));
                    }
                    uint32_t e0, o0, e1, o1;
                    f8x4_to_f16(kw.x, e0, o0);
                    f8x4_to_f16(kw.y, e1, o1);
                    reinterpret_cast<uint2*>(kq + (size_t)(r - w0) * D)[lane] =
                        make_uint2(__byte_perm(h2_q8(e0, inv2), h2_q8(o0, inv2), 0x6240), __byte_perm(h2_q8(e1, inv2), h2_q8(o1, inv2), 0x6240));
                    f8x4_to_f16(vw.x, e0, o0);
                    f8x4_to_f16(vw.y, e1, o1);
                    vh[(size_t)(r - w0) * 32 + lane] = make_uint4(e0, o0, e1, o1);
                }
                __syncthreads();
            }
            if (tid == 0) {
                __threadfence();
                atomicAdd(ctr + kPkpCtrConv + wi, 1);
            }
            if (tr) tr[6] = pkp_now(), tr[7] = 2 | (wi << 8);
            if (tid == 0) {
                release();
                s_item = QE_PKP_PREGRAB ? nxt : atomicAdd(ctr + kPkpCtrNext, 1);
            }
            continue;
        }
        // ---- ATTN(wi)
        const int a = j - nC;
        const int qt = a % QT, sp = (a / QT) % S, hh = a / (QT * S);
        int h = 0;
        {
            int kvo[kMaxLocalKv];
            for (int jj = 0; jj < s.nkv; jj++) kvo[jj] = jj;
            for (int x = 1; x < s.nkv; x++)
                for (int y = x; y > 0; y--) {
                    const PosMap& px = s.kv_pm[kvo[y]];
                    const PosMap& py = s.kv_pm[kvo[y - 1]];
                    if ((long long)px.A * py.P > (long long)py.A * px.P) { const int tt = kvo[y]; kvo[y] = kvo[y - 1]; kvo[y - 1] = tt; }
                    else break;
                }
            int rem = hh;
            for (int jj = 0; jj < s.nkv; jj++) {
                const int kj = kvo[jj], n = s.kv_qe[kj] - s.kv_qb[kj];
                if (rem < n) { h = s.kv_qb[kj] + rem; break; }
                rem -= n;
            }
        }
        const int q0 = qt * kRows, kv = s.q2kv[h];
        const int G = min(kRows, m - q0);
        const PosMap pm = s.kv_pm[kv];
        const int kendb = pm_rank(pm, p0 + min(q0 + kRows, m));
        // split sp of window v: an even share of the window's rows of this kv head (64-key multiples);
        // midx = merges into the slot before this item (its non-empty items of earlier windows)
        const int krows = pm_rank(pm, p0 + m);
        auto wrange = [&](int v, int& b0, int& b1) {
            const int vr = min(W, krows - v * W);
            int ch = (max(vr, 0) + S - 1) / S;
            ch = (ch + kKeys - 1) / kKeys * kKeys;
            b0 = v * W + sp * ch;
            b1 = min(min(v * W + W, kendb), b0 + ch);
        };
        int kbeg, kend;
        wrange(wi, kbeg, kend);
        int midx = 0;
        for (int v = 0; v < wi; v++) {
            int b0, b1;
            wrange(v, b0, b1);
            midx += b0 < b1;
        }
        int* sflag = ctr + kPkpCtrSlot + (sp * s.nq + h) * QT + qt;
        auto pacc = [&](int row) { return work + (((size_t)sp * m + q0 + row) * s.nq + h) * 256; };
        auto pml = [&](int row) {
            return work + (size_t)S * m * s.nq * 256 + (((size_t)sp * m + q0 + row) * s.nq + h) * 2;
        };
        if (kbeg >= kend) {  // no keys here: window 0 marks the slot empty if no window has keys for it
            bool any = false;
            if (wi == 0)
                for (int v = 1; v < nwin && !any; v++) {
                    int b0, b1;
                    wrange(v, b0, b1);
                    any = b0 < b1;
                }
            if (wi == 0 && !any)
                for (int r = tid; r < G; r += NT) {
                    pml(r)[0] = -FLT_MAX;
                    pml(r)[1] = 0.f;
                }
            if (tid == 0) atomicAdd(ctr + kPkpCtrAttn + wi, 1);
            if (tr) tr[6] = pkp_now(), tr[7] = 4 | (wi << 8);
            if (tid == 0) {
                release();
                s_item = QE_PKP_PREGRAB ? nxt : atomicAdd(ctr + kPkpCtrNext, 1);
            }
            continue;
        }
        if (tid == 0) {
            if (conv_ok != wi) pkp_wait_ge(ctr + kPkpCtrConv + wi, nC, 2000 + wi), conv_ok = wi;
            if (!q_ok) pkp_wait_ge(ctr + kPkpCtrQ, nQ, 3000), q_ok = true;
        }
        if (tr) tr[3] = pkp_now();
        __syncthreads();
        uint32_t qf[D / 32][4];
        float sq0, sq1;
        const int r0 = warp * 16 + g;
        if (nQ) {
            const uint4* src = ps.qf + ((size_t)(qt * s.nq + h) * 8 + warp) * (D / 32) * 32;
#pragma unroll
            for (int kk = 0; kk < D / 32; kk++) {
                const uint4 v = __ldcg(src + kk * 32 + lane);
                qf[kk][0] = v.x; qf[kk][1] = v.y; qf[kk][2] = v.z; qf[kk][3] = v.w;
            }
            const float* qs = ps.qsq + (size_t)(qt * s.nq + h) * kRows;
            sq0 = __ldcg(qs + r0);
            sq1 = __ldcg(qs + r0 + 8);
        } else {
            quant_q(qt, h, qf);
            sq0 = s_sq[r0];
            sq1 = s_sq[r0 + 8];
        }

        const uint8_t* kqb = ps.kq[b][kv];
        const uint4* vhb = ps.vh[b][kv];
        const uint2* kscr = s.ksc + (size_t)kv * s.ksc_ld;
        const int cr = tid >> 4, cc = tid & 15, vr = tid >> 5, vcl = tid & 31;
        const int wsw8 = cr * 16 + (cc ^ (cr & 7));
        const int ntiles = (kend - kbeg + kKeys - 1) / kKeys;
        if (g_pkp_diag && tid == 0) g_pkp_diag[3100 + blockIdx.x] = kbeg, g_pkp_diag[3200 + blockIdx.x] = kend, g_pkp_diag[3300 + blockIdx.x] = midx;
        auto load = [&](int tile, int buf) {
            const int k0 = kbeg + tile * kKeys;
            uint4* kb = sm + buf * kPKBuf;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int key = k0 + cr + 16 * u;
                cp_async16m(kb + wsw8 + 256 * u, kqb + (size_t)(min(key, kend - 1) - w0) * D + cc * 16, key < kend);
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const int r = vr + 8 * u, key = k0 + r;
                cp_async16m(kb + 1024 + sw(r, vcl), vhb + (size_t)(min(key, kend - 1) - w0) * 32 + vcl, key < kend);
            }
            if (tid < kKeys) {
                const float skq = __uint_as_float(__ldcg(&kscr[min(k0 + tid, kend - 1)].y));
                reinterpret_cast<float2*>(sm + kPKSK)[buf * kKeys + tid] = make_float2(skq, -(12582912.f * skq));
            }
        };
        const int mi = lane >> 3, L7 = lane & 7;
        unsigned kq4[4], vq4[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            kq4[i] = sbase + 16u * (((mi >> 1) * 8 + L7) * 16 + ((2 * i + (mi & 1)) ^ L7));
            vq4[i] = sbase + 16u * (1024 + ((mi & 1) * 8 + L7) * 32 + ((2 * i + (mi >> 1)) ^ L7));
        }
        load(0, 0);
        cp_commit();

        float o[32][4];
#pragma unroll
        for (int jj = 0; jj < 32; jj++) o[jj][0] = o[jj][1] = o[jj][2] = o[jj][3] = 0.f;
        float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
        const int qpos0 = p0 + q0 + min(r0, G - 1), qpos1 = p0 + q0 + min(r0 + 8, G - 1);
        const int rl0 = pm_rank(pm, p0 + q0);
        const float vscale = s.vs * 256.f;
        constexpr uint32_t kOnes = 0x3C003C00u;

        for (int tile = 0; tile < ntiles; tile++) {
            const int k0 = kbeg + tile * kKeys, bb2 = tile & 1;
            cp_wait<0>();
            __syncthreads();  // tile ready everywhere; tile - 1's buffer free
            if (tile + 1 < ntiles) load(tile + 1, bb2 ^ 1);
            cp_commit();
            if (tile == 1 && tid == 0) release();
            const unsigned bo = (unsigned)(bb2 * kPKBuf * 16);
            uint32_t si[NJ][4];
#pragma unroll
            for (int jj = 0; jj < NJ; jj++) si[jj][0] = si[jj][1] = si[jj][2] = si[jj][3] = 0x4B400000u;
#pragma unroll
            for (int kk = 0; kk < D / 32; kk++)
#pragma unroll
                for (int jp = 0; jp < 4; jp++) {
                    uint32_t bb[4];
                    ldsm_x4a(bb, kq4[kk & 3] + bo + jp * 4096 + (kk >> 2) * 128);
                    imma(si[jp * 2], qf[kk], bb[0], bb[1]);
                    imma(si[jp * 2 + 1], qf[kk], bb[2], bb[3]);
                }
            float sf[NJ][4];
            const float4* skp = reinterpret_cast<const float4*>(sm + kPKSK) + bb2 * (kKeys / 2);
#pragma unroll
            for (int jj = 0; jj < NJ; jj++) {
                const float4 k4 = skp[jj * 4 + t];
                sf[jj][0] = fmaf(__uint_as_float(si[jj][0]), k4.x, k4.y);
                sf[jj][1] = fmaf(__uint_as_float(si[jj][1]), k4.z, k4.w);
                sf[jj][2] = fmaf(__uint_as_float(si[jj][2]), k4.x, k4.y);
                sf[jj][3] = fmaf(__uint_as_float(si[jj][3]), k4.z, k4.w);
            }
            if (k0 + kKeys > min(kend, rl0)) {
#pragma unroll
                for (int jj = 0; jj < NJ; jj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = k0 + jj * 8 + 2 * t + (e & 1);
                        if (key >= kend || pm_unrank(pm, key) > ((e < 2) ? qpos0 : qpos1)) sf[jj][e] = -INFINITY;
                    }
            }
            uint32_t sc[NJ][2];
            float alpha[2];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                float tmax = -INFINITY;
#pragma unroll
                for (int jj = 0; jj < NJ; jj++) tmax = fmaxf(tmax, fmaxf(sf[jj][2 * r], sf[jj][2 * r + 1]));
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
                const float qs = r ? sq1 : sq0;
                const float nm = fmaxf(mx[r], tmax * qs);
                alpha[r] = mx[r] == -INFINITY ? 0.f : exp2f(mx[r] - nm);
                mx[r] = nm;
                const float mm = nm == -INFINITY ? 0.f : nm;
#pragma unroll
                for (int jj = 0; jj < NJ; jj++)
                    sc[jj][r] = pack_f16(exp2f(fmaf(sf[jj][2 * r], qs, -mm)), exp2f(fmaf(sf[jj][2 * r + 1], qs, -mm)));
            }
            {  // row sums of P in fp16 over 32 keys (as P V), the halves added in fp32 (QE_PKP_L32=0: over 64 keys)
                uint32_t cl[2] = {0u, 0u}, dl[2] = {0u, 0u};
#pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    const uint32_t af[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                    if (QE_PKP_L32 && ks >= 2) mma_hacc(dl, af, kOnes, kOnes);
                    else mma_hacc(cl, af, kOnes, kOnes);
                }
                l[0] = l[0] * alpha[0] + (__low2float(*reinterpret_cast<__half2*>(&cl[0])) + __low2float(*reinterpret_cast<__half2*>(&dl[0])));
                l[1] = l[1] * alpha[1] + (__low2float(*reinterpret_cast<__half2*>(&cl[1])) + __low2float(*reinterpret_cast<__half2*>(&dl[1])));
            }
#pragma unroll
            for (int jp = 0; jp < 16; jp++) {
                uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u}, d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    const uint32_t af[4] = {sc[2 * ks][0], sc[2 * ks][1], sc[2 * ks + 1][0], sc[2 * ks + 1][1]};
                    uint32_t bb[4];
                    ldsm_x4ta(bb, vq4[jp & 3] + bo + ks * 8192 + (jp >> 2) * 128);
                    if (ks >= 2) {
                        mma_hacc(d0, af, bb[0], bb[1]);
                        mma_hacc(d1, af, bb[2], bb[3]);
                    } else {
                        mma_hacc(c0, af, bb[0], bb[1]);
                        mma_hacc(c1, af, bb[2], bb[3]);
                    }
                }
                const float2 a0 = __half22float2(*reinterpret_cast<__half2*>(&c0[0])), b0 = __half22float2(*reinterpret_cast<__half2*>(&d0[0]));
                const float2 a1 = __half22float2(*reinterpret_cast<__half2*>(&c0[1])), b1 = __half22float2(*reinterpret_cast<__half2*>(&d0[1]));
                const float2 a2 = __half22float2(*reinterpret_cast<__half2*>(&c1[0])), b2 = __half22float2(*reinterpret_cast<__half2*>(&d1[0]));
                const float2 a3 = __half22float2(*reinterpret_cast<__half2*>(&c1[1])), b3 = __half22float2(*reinterpret_cast<__half2*>(&d1[1]));
                float* pa = o[jp * 2];
                float* pb = o[jp * 2 + 1];
                pa[0] = pa[0] * alpha[0] + (a0.x + b0.x); pa[1] = pa[1] * alpha[0] + (a0.y + b0.y);
                pa[2] = pa[2] * alpha[1] + (a1.x + b1.x); pa[3] = pa[3] * alpha[1] + (a1.y + b1.y);
                pb[0] = pb[0] * alpha[0] + (a2.x + b2.x); pb[1] = pb[1] * alpha[0] + (a2.y + b2.y);
                pb[2] = pb[2] * alpha[1] + (a3.x + b3.x); pb[3] = pb[3] * alpha[1] + (a3.y + b3.y);
            }
        }
        cp_wait<0>();
        __syncthreads();  // every read of the window buffer done (and smem free for the next item)
        if (tid == 0) {
            release();  // (ntiles == 1)
            if (tr) tr[4] = pkp_now();
            atomicAdd(ctr + kPkpCtrAttn + wi, 1);
            if (midx > 0) pkp_wait_ge(sflag, midx, 4000 + wi);  // the slot holds the earlier windows' merges
            if (tr) tr[5] = pkp_now();
        }
        __syncthreads();

        // ---- partial of this window: window 0 writes, later windows merge into the slot (in window order).
        // The slot's old rows are staged in smem with cp.async (64 rows per phase, all copies in flight):
        // register loads of them were latency-bound (~40 us per item at 16K windows).
        const bool odd = t & 1;
        const int dbase = 4 * (t >> 1) + (odd ? 2 : 0);
        constexpr int kStLd = 264;  // staged row stride in floats (1 KB + 32 B: rows on different banks)
        float* stage = reinterpret_cast<float*>(sm);
        for (int ph = 0; ph < 2; ph++) {
            if (midx > 0) {
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const int c = tid + NT * i, row = c >> 6, col = c & 63, gr = 64 * ph + row;
                    cp_async16m(stage + row * kStLd + col * 4, pacc(min(gr, G - 1)) + col * 4, gr < G);
                }
                cp_commit();
                cp_wait<0>();
            }
            __syncthreads();  // staged rows ready (and the previous phase's stores read out of smem)
            if (tr) tr[8 + 3 * ph] = pkp_now();
            if ((warp >> 2) == ph) {
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int row = r0 + 8 * r;
                    float v[32][2];
#pragma unroll
                    for (int jj = 0; jj < 32; jj++) {
                        const float x0 = o[jj][2 * r], x1 = o[jj][2 * r + 1];
                        const float rcv = __shfl_xor_sync(0xffffffffu, odd ? x0 : x1, 1);
                        v[jj][0] = odd ? rcv : x0;
                        v[jj][1] = odd ? x1 : rcv;
                    }
                    // lanes of rows >= G (G % 8 != 0) skip: the warp sync below names only the others
                    const unsigned live = __ballot_sync(0xffffffffu, row < G);
                    if (row >= G) continue;
                    const float mn = mx[r] == -INFINITY ? -FLT_MAX : mx[r] * 0.6931471805599453f, ln = l[r];
                    float fo = 0.f, fn = 1.f, mo = -FLT_MAX, lo = 0.f;
                    if (midx > 0) {
                        mo = __ldcg(pml(row));
                        lo = __ldcg(pml(row) + 1);
                        const float M = fmaxf(mo, mn);
                        fo = (mo == -FLT_MAX || lo == 0.f) ? 0.f : __expf(mo - M);
                        fn = (mn == -FLT_MAX || ln == 0.f) ? 0.f : __expf(mn - M);
                    }
                    float* so = stage + (row - 64 * ph) * kStLd + dbase;
#pragma unroll
                    for (int jj = 0; jj < 32; jj++) {
                        float2 nv = make_float2(v[jj][0] * vscale * fn, v[jj][1] * vscale * fn);
                        if (midx > 0 && fo != 0.f) {
                            const float2 ov = *reinterpret_cast<const float2*>(so + jj * 8);
                            nv.x += ov.x * fo;
                            nv.y += ov.y * fo;
                        }
                        *reinterpret_cast<float2*>(so + jj * 8) = nv;
                    }
                    __syncwarp(live);  // the row's lanes read the old (m, l) before lane t == 0 overwrites it
                    if (t == 0) {
                        if (midx == 0) {
                            pml(row)[0] = mn;
                            pml(row)[1] = ln;
                        } else {
                            const float M = fmaxf(mo, mn);
                            pml(row)[0] = (fo == 0.f && fn == 0.f) ? -FLT_MAX : M;
                            pml(row)[1] = lo * fo + ln * fn;
                        }
                    }
                }
            }
            __syncthreads();  // merged rows in smem -> coalesced 16-byte stores
            if (tr) tr[9 + 3 * ph] = pkp_now();
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int c = tid + NT * i, row = c >> 6, col = c & 63, gr = 64 * ph + row;
                if (gr < G) *reinterpret_cast<uint4*>(pacc(gr) + col * 4) = *reinterpret_cast<const uint4*>(stage + row * kStLd + col * 4);
            }
            if (tr) tr[10 + 3 * ph] = pkp_now();
        }
        __syncthreads();
        if (tid == 0) rel_p = sflag, rel_v = midx + 1;
        if (tr) tr[6] = pkp_now(), tr[7] = 3 | (wi << 8) | (ntiles << 16);
        if (tid == 0) s_item = QE_PKP_PREGRAB ? nxt : atomicAdd(ctr + kPkpCtrNext, 1);
    }
}

}  // namespace
__global__ void qe_mhz_kernel(unsigned long long* out) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    const long long c0 = clock64();
    while (clock64() - c0 < 2000000) {}
    const long long c1 = clock64();
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1));
    out[0] = (unsigned long long)(c1 - c0);
    out[1] = t1 - t0;
}
// bench: SM clock (MHz) over ~1 ms of spinning
double qe_sm_mhz_dev() {
    unsigned long long* d;
    unsigned long long h[2];
    CUDA_CHECK(cudaMalloc(&d, 16));
    qe_mhz_kernel<<<1, 1>>>(d);
    CUDA_CHECK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    cudaFree(d);
    return (double)h[0] / (double)h[1] * 1e3;
}
// bench only: per-item trace of attn_pkp_kernel into buf (8 x u64 per queue item, null: off)
void qe_pkp_trace(unsigned long long* buf) { CUDA_CHECK(cudaMemcpyToSymbol(g_pkp_trace, &buf, sizeof(buf))); }
namespace {

// Prefill stats exchange, grid ceil(m * nx / 256), block 256: thread per record (row, exchange q
// head): (max, sum) over the split-K partials of a position-split head's q head go to the peer
// (see attn_stats_dev); the last block of the launch raises the peer's flags on every exchange.
__global__ void prefill_stats_kernel(const float* __restrict__ work, int splits, int m, const AttnShape s,
                                     const int* __restrict__ pos) {
    __shared__ int xh[32];
    __shared__ int s_nx;
    if (threadIdx.x == 0) s_nx = xchg_heads(s, xh);
    __syncthreads();
    const int nx = s_nx, nq = s.nq, p0 = pos[0];
    const float* ml = work + (size_t)splits * m * nq * 256;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m * nx; r += gridDim.x * blockDim.x) {
        const int qi = r / nx, h = xh[r % nx], kv = s.q2kv[h];
        if (!xchg_has(s, kv, p0, m, 0)) continue;
        float mx = -FLT_MAX, den = 0.f;
        for (int sp = 0; sp < splits; sp++) mx = fmaxf(mx, ml[(((size_t)sp * m + qi) * nq + h) * 2]);
        for (int sp = 0; sp < splits; sp++) {
            const size_t b = ((size_t)sp * m + qi) * nq + h;
            const float l = ml[b * 2 + 1];
            if (l != 0.f) den += __expf(ml[b * 2] - mx) * l;
        }
        float* dst = s.xpo[kv] + ((size_t)(s.xrow0 + qi) * (s.kv_qe[kv] - s.kv_qb[kv]) + (h - s.kv_qb[kv])) * kXchgFloats;
        *reinterpret_cast<float2*>(dst) = make_float2(mx, den);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(s.xdone, 1u) == gridDim.x - 1) {
        *s.xdone = 0;
        __threadfence();
        for (int j = 0; j < s.nkv; j++)
            if (s.kv_x[j] == 3) qe_st_release_sys(s.xpflag[j], xchg_seq(s));
    }
}

// Prefill combine, grid (m, nq), block 256: merges the split-K partials of a head; a position-split
// head waits for the peer's stats and scales its numerator by e^(m - M) / D (global M, D), then
// sigmoid(gate), bf16 output. Heads written by the attention kernel itself (whole head, one
// split) are left alone.
__global__ void prefill_combine_kernel(const float* __restrict__ work, int splits, int m, const AttnShape s,
                                       const float* __restrict__ proj, int ld, bf16* __restrict__ out,
                                       const int* __restrict__ pos) {
    const int qi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int nq = s.nq, kv = s.q2kv[h], no = s.no ? s.no : s.nq;
    const bool xch = s.kv_x[kv] == 3;
    if (!xch && splits == 1) return;
    if (h < s.ob || h >= s.ob + no) return;
    const int p0 = pos[0];
    const float* ml = work + (size_t)splits * m * nq * 256;
    float mx = -FLT_MAX;
    for (int sp = 0; sp < splits; sp++) mx = fmaxf(mx, ml[(((size_t)sp * m + qi) * nq + h) * 2]);
    float num = 0.f, den = 0.f;
    for (int sp = 0; sp < splits; sp++) {
        const size_t b = ((size_t)sp * m + qi) * nq + h;
        const float l = ml[b * 2 + 1];
        if (l == 0.f) continue;
        const float w = __expf(ml[b * 2] - mx);
        num += w * work[b * 256 + d];
        den += w * l;
    }
    float v;
    if (xch) {  // the peer's flag, raised on every exchange (data or not)
        if (d == 0) {
            unsigned ws = 0;
            while ((int)(qe_ld_acquire_sys(s.xflag[kv]) - xchg_seq(s)) < 0) {
                __nanosleep(64);
                // a stuck wait (peer gone / flag-sequence mismatch) is reported by the watchdog, not silent
                QE_WD_POLL(ws, kWdKvPrefill, -1, xchg_seq(s), qe_ld_acquire_sys(s.xflag[kv]));
            }
        }
        __syncthreads();
    }
    if (xch && xchg_needed(s, kv, p0, m, 0)) {  // the peer holds keys of this head: merge its stats
        __shared__ float s_rm, s_rl;
        if (d == 0) {
            const float* rem = s.xo[kv] + ((size_t)(s.xrow0 + qi) * (s.kv_qe[kv] - s.kv_qb[kv]) + (h - s.kv_qb[kv])) * kXchgFloats;
            const float2 r = __ldcg(reinterpret_cast<const float2*>(rem));
            s_rm = r.x;
            s_rl = r.y;
        }
        __syncthreads();
        const float rm = s_rm, rl = s_rl;
        const float M = fmaxf(mx, rm);
        const float wa = mx == -FLT_MAX ? 0.f : __expf(mx - M), wb = rm == -FLT_MAX ? 0.f : __expf(rm - M);
        const float D = den * wa + rl * wb;
        v = den == 0.f ? 0.f : num * wa / D;
    } else {
        v = num / den;
    }
    const float gate = proj[(size_t)qi * ld + h * 512 + 256 + d];
    out[((size_t)qi * no + (h - s.ob)) * 256 + d] = __float2bfloat16(v / (1.f + __expf(-gate)));
}

}  // namespace

size_t attention_prefill_work_floats(int max_rows_x_splits, int nq) {
    return (size_t)max_rows_x_splits * nq * (256 + 2);
}

template <bool QK8, int NW>
static void set_pf8() {
    CUDA_CHECK(cudaFuncSetAttribute(attn_pf8_kernel<QK8, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)p8_smem<QK8, NW * 8>()));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pf8_kernel<QK8, NW>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
}
// QE_PATTN_NW: warps of attn_pf8_kernel (8 = 128 rows, 64-key tiles, 1 block/SM; 4 = 64 rows, 32-key tiles, 2 blocks/SM)
static bool pf8_pp() {  // QE_PATTN_PP=1: the ping-pong kernel attn_pp8_kernel (fp16 Q K^T only)
    static const bool pp = env_set("QE_PATTN_PP") ? env_int("QE_PATTN_PP", false) != 0 : false;
    return pp;
}
static int pf8_nw() {
    static const int nw = env_int("QE_PATTN_NW", 8);
    return nw == 4 ? 4 : 8;
}
template <bool QK8, int NW>
static void launch_pf8(int m, int splits, const bf16* q, const void* kc, const void* vc, const AttnShape& s2,
                       const float* proj, int ld, const int* pos, bf16* out, float* work, cudaStream_t st) {
    const int nb = ((m + NW * 16 - 1) / (NW * 16)) * s2.nq * splits;
    attn_pf8_kernel<QK8, NW><<<nb, NW * 32, p8_smem<QK8, NW * 8>(), st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, s2,
                                                                         proj, ld, pos, m, out, work);
}

void attention_prefill_init() {
    set_attr<4, false, false>();
    set_attr<4, true, false>();
    CUDA_CHECK(cudaFuncSetAttribute(attn_pp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kPPSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pp8_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kPKSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pkv_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pkp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kPKSmem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pkp_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    {
        const int cpos = env_int("QE_PKP_CPOS", 50);
        CUDA_CHECK(cudaMemcpyToSymbol(s_pkp_cpos, &cpos, sizeof(int)));
    }
    if (env_str("QE_PKP_DIAG")) {  // debug: host-mapped wait state of attn_pkp_kernel + a watchdog thread
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int* h = nullptr;
        CUDA_CHECK(cudaHostAlloc((void**)&h, 4096 * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        memset(h, 0, 4096 * sizeof(int));
        int* d = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d, h, 0));
        CUDA_CHECK(cudaMemcpyToSymbol(g_pkp_diag, &d, sizeof(d)));
        std::thread([h, dev]() {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(15));
                volatile int* v = h;
                fprintf(stderr, "[pkp diag] dev %d block starts %d ends %d\n", dev, v[2000], v[2001]);
                for (int b = 0; b < 128; b++)
                    if (v[2002 + b]) fprintf(stderr, "[pkp diag] dev %d block %d running, item %d kbeg %d kend %d midx %d tile %d\n", dev, b, v[3000 + b], v[3100 + b], v[3200 + b], v[3300 + b], v[3400 + b]);
                fprintf(stderr, "[pkp diag] dev %d call m %d p0 %d nwin %d S %d W %d total %d nQ %d nC %d nA %d\n", dev, v[1990], v[1991], v[1992], v[1993], v[1994], v[1995], v[1996], v[1997], v[1998]);
                for (int b = 0; b < 128; b++)
                    if (v[b * 8 + 4]) fprintf(stderr, "[pkp diag] dev %d block %d tag %d target %d value %d addr %x\n", dev, b, v[b * 8], v[b * 8 + 1], v[b * 8 + 2], v[b * 8 + 3]);
            }
        }).detach();
    }
    set_pf8<false, 8>();
    set_pf8<true, 8>();
    set_pf8<false, 4>();
    set_pf8<true, 4>();
}

// QE_PATTN_V2=0: the 4-warp attn_tc_kernel for fp8 prefill (before attn_pf8_kernel); also used for m < 128
static bool pattn_v2(const AttnShape& s, int m) {
    static const int v2 = env_int("QE_PATTN_V2", 1);
    return v2 && s.fp8 && m >= 128;
}

template <int MT>
void set_dec8d() {
    CUDA_CHECK(cudaFuncSetAttribute(attn_dec8d_kernel<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)dec8d_smem<MT>()));
    CUDA_CHECK(cudaFuncSetAttribute(attn_dec8d_seg_kernel<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)dec8d_smem<MT>()));
}

void attention_decode_tc_init() {
    set_dec8d<1>();
    set_dec8d<2>();
    set_dec8d<3>();
    set_dec8<1, kDecKG>();
    set_dec8<2, kDecKG>();
    set_dec8<3, kDecKG>();
    set_attr<1, false, true>();
    set_attr<2, false, true>();
    set_attr<3, false, true>();
    set_attr<1, true, true>();
    set_attr<2, true, true>();
    set_attr<3, true, true>();
}

// int8 Q K^T (exact int32 scores, fp32 softmax) everywhere by default: with the 32-key fp16 P V it
// has the lowest KL vs fp32 attention of all variants, short and long prompts. QE_PATTN_QK8_MIN = n > 0: only for calls whose queries start at >= n (fp16 Q K^T below);
// QE_PATTN_QK8=1 / 0 forces it on / off everywhere.
static bool pattn_qk8(const AttnShape& s) {
    static const int qk8_env = env_int("QE_PATTN_QK8", -1);
    static const int qk8_min = env_int("QE_PATTN_QK8_MIN", 0);
    return qk8_env >= 0 ? qk8_env != 0 : (qk8_min <= 0 || (s.pf_p0 >= 0 && s.pf_p0 >= qk8_min));
}
// K smoothing (k - mean over the call, + q . mean exact): off - short-prompt KL 7,4e-4 vs 7,05e-4 without
static bool pattn_smooth() {
    static const bool v = env_int("QE_PATTN_KSMOOTH", 0);
    return v;
}
// windowed precomputed K / V: QE_PATTN_KVPRE = 2 (default) persistent kernel attn_pkp_kernel, 1 = one
// conversion + attention launch pair per window (attn_pkv_kernel), 0 = off
int g_pattn_kvpre = -1, g_pkp_splits = -1, g_pkp_min = -1;  // bench overrides (qe_pattn_mode), -1: env / default
static int pattn_kvpre() {
    static const int v = env_int("QE_PATTN_KVPRE", 2);
    return g_pattn_kvpre >= 0 ? g_pattn_kvpre : v;
}
// the per-key int8 K scale scratch holds every local key row the call can reach
static bool pattn_ksc_fits(const AttnShape& s) {
    bool fits = s.ksc != nullptr;
    for (int j = 0; j < s.nkv; j++) fits &= pm_cap(s.kv_pm[j], s.split + s.max_t) <= s.ksc_ld;
    return fits;
}
// the persistent windowed attention takes this call (its split count comes from pkp_splits)
static bool pkp_on(const AttnShape& s, int m) {
    if (!pattn_v2(s, m) || pattn_kvpre() != 2 || !s.pscr || s.pscr->w <= 0 || !s.pscr->ctr || s.pf_p0 < 0) return false;
    if (!pattn_qk8(s) || pattn_smooth() || !pattn_ksc_fits(s)) return false;
    // QE_PKP_MIN (default 16384): calls whose queries end below it keep attn_pf8, bit for bit as master (e2e gain
    // below 16K is within noise)
    static const int pmin = env_int("QE_PKP_MIN", 16384);
    if (s.pf_p0 + m < (g_pkp_min >= 0 ? g_pkp_min : pmin)) return false;
    int rows = 0;
    for (int j = 0; j < s.nkv; j++) rows = std::max(rows, pm_rank(s.kv_pm[j], s.pf_p0 + m));
    return (rows + s.pscr->w - 1) / s.pscr->w <= kPkpMaxWin;
}
// Makespan model of attn_pkp_kernel's attention items for S splits per window: list scheduling in queue order
// on nsm SMs, item = tiles x t_tile + a fixed cost (Q fragments, merge epilogue), empty items ~free
static double pkp_model(const AttnShape& s, int m, int S, int nsm) {
    constexpr double kTile = 5.05, kItem = 15.0;
    const int W = s.pscr->w, p0 = s.pf_p0, QT = (m + 127) / 128;
    int rows = 0;
    for (int j = 0; j < s.nkv; j++) rows = std::max(rows, pm_rank(s.kv_pm[j], p0 + m));
    const int nwin = (rows + W - 1) / W;
    // heads in the kernel's (LPT) order
    int kvo[kMaxLocalKv];
    for (int j = 0; j < s.nkv; j++) kvo[j] = j;
    std::stable_sort(kvo, kvo + s.nkv, [&](int x, int y) {
        return (long long)s.kv_pm[x].A * s.kv_pm[y].P > (long long)s.kv_pm[y].A * s.kv_pm[x].P;
    });
    std::vector<double> sm(nsm, 0.0);  // min-heap of free times
    std::make_heap(sm.begin(), sm.end(), std::greater<double>());
    for (int w = 0; w < nwin; w++)
        for (int jj = 0; jj < s.nkv; jj++) {
            const int kv = kvo[jj];
            const PosMap& pm = s.kv_pm[kv];
            const int krows = pm_rank(pm, p0 + m);
            const int vr = std::min(W, krows - w * W);
            int ch = (std::max(vr, 0) + S - 1) / S;
            ch = (ch + 63) / 64 * 64;
            for (int hq = s.kv_qb[kv]; hq < s.kv_qe[kv]; hq++)
                for (int sp = 0; sp < S; sp++)
                    for (int qt = 0; qt < QT; qt++) {
                        const int kendb = pm_rank(pm, p0 + std::min(qt * 128 + 128, m));
                        const int b0 = w * W + sp * ch, b1 = std::min(std::min(w * W + W, kendb), b0 + ch);
                        if (b0 >= b1) continue;
                        std::pop_heap(sm.begin(), sm.end(), std::greater<double>());
                        sm.back() += ((b1 - b0 + 63) / 64) * kTile + kItem;
                        std::push_heap(sm.begin(), sm.end(), std::greater<double>());
                    }
        }
    return *std::max_element(sm.begin(), sm.end());
}
// split slots of the persistent kernel per window (>= 2: the combine always runs): QE_PKP_SPLITS, else the
// best of 2..5 by pkp_model (cached for the last call shape: the layers of a micro-batch repeat it)
static int pkp_splits(const AttnShape& s, int m, int smax) {
    static const int v = env_int("QE_PKP_SPLITS", 0);
    const int f = g_pkp_splits > 0 ? g_pkp_splits : v;
    if (f > 0) return std::max(2, std::min(f, smax));
    static thread_local long long key = -1;
    static thread_local int best = 2;
    long long k = ((long long)s.pf_p0 * 4096 + m) * 64 + smax;
    for (int j = 0; j < s.nkv; j++) k = k * 1315423911LL + s.kv_pm[j].A * 7919 + s.kv_pm[j].P * 31 + s.kv_pm[j].O;
    if (k == key) return best;
    int dev = 0, nsm = 82;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    double bt = 1e300;
    for (int S = 2; S <= smax; S++) {
        const double t = pkp_model(s, m, S, nsm);
        if (t < bt * 0.995) bt = t, best = S;
    }
    key = k;
    return best;
}

int attention_prefill_splits(const AttnShape& s, int m, size_t work_floats, bool have_work) {
    const bool v2 = pattn_v2(s, m);
    const int prow = v2 ? (pf8_pp() ? 128 : pf8_nw() * 16) : 64;
    const int qt = (m + prow - 1) / prow;
    // split the key range so the blocks fill whole waves of the slots (82 SMs x 2 blocks, v2: x 1): e.g.
    // m = 1024 -> 128 tiles (78% of one wave) -> 5 splits = 640 blocks = 3.9 waves (+22..25%)
    const int tiles = qt * s.nq, slots = v2 && (pf8_nw() == 8 || pf8_pp()) ? 82 : 164;
    int splits = 1;
    double best = 1e30;
    for (int sp = 1; sp <= std::max(12, slots / tiles); sp++) {
        if (sp > 1 && attention_prefill_work_floats(sp * m, s.nq) > work_floats) break;
        const double waves = (double)((tiles * sp + slots - 1) / slots);
        const double c = waves / sp * (sp > 1 ? 1.02 : 1.0);  // + partials and combine
        if (c < best - 1e-9) best = c, splits = sp;
    }
    static const int force = env_int("QE_PATTN_SPLITS", 0);
    if (force) splits = force;
    while (splits > 1 && attention_prefill_work_floats(splits * m, s.nq) > work_floats) splits--;
    if (!have_work) splits = 1;
    if (splits >= 2 && pkp_on(s, m)) {
        const int QT = (m + 127) / 128;
        int smax = 5;
        while (smax > 2 && (attention_prefill_work_floats(smax * m, s.nq) > work_floats || kPkpCtrSlot + QT * s.nq * smax > kPkpCtrInts))
            smax--;
        if (attention_prefill_work_floats(smax * m, s.nq) <= work_floats && kPkpCtrSlot + QT * s.nq * smax <= kPkpCtrInts)
            splits = pkp_splits(s, m, smax);
    }
    return splits;
}

void attention_prefill_kernel(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const float* proj,
                              int ld, const int* pos, int m, bf16* out, cudaStream_t st, float* work, int splits) {
    if (pattn_v2(s, m)) {
        AttnShape s2 = s;
        s2.pf_splits = splits;
        const bool qk8 = pattn_qk8(s);
        if (!qk8 || !s.ksc) s2.kmu = nullptr;
        if (qk8 && s.ksc) {
            const bool fits = pattn_ksc_fits(s);
            const bool smooth = pattn_smooth();
            if (fits && smooth && s.kmu) {
                CUDA_CHECK(cudaMemsetAsync(s.kmu, 0, (size_t)s.nkv * D * sizeof(float), st));
                pattn_kmean_kernel<<<dim3(164, s.nkv), 256, 0, st>>>((const uint8_t*)kc, s, pos, m, s.kmu);
            } else {
                s2.kmu = nullptr;
            }
            // persistent windowed attention (the per-key K scales come from its conversion items)
            if (splits >= 2 && pkp_on(s, m)) {
                PattnScratch ps = *s.pscr;
                int rows = 0;
                for (int j = 0; j < s.nkv; j++) rows = std::max(rows, pm_rank(s.kv_pm[j], s.pf_p0 + m));
                const int nwin = (rows + ps.w - 1) / ps.w, QT = (m + 127) / 128;
                if (QT > ps.qt_max) ps.qf = nullptr, ps.qsq = nullptr;
                CHECK(kPkpCtrSlot + QT * s.nq * splits <= kPkpCtrInts, "pkp slots");
                int dev = 0, nsm = 0;
                CUDA_CHECK(cudaGetDevice(&dev));
                CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
                CUDA_CHECK(cudaMemsetAsync(ps.ctr, 0, (size_t)(kPkpCtrSlot + QT * s.nq * splits) * sizeof(int), st));
                attn_pkp_kernel<<<nsm, 256, kPKSmem, st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, s2, ps, pos, m, work, nwin);
                return;
            }
            if (fits) pattn_kscale_kernel<<<dim3(164, s.nkv), 256, 0, st>>>((const uint8_t*)kc, s2, pos, m, s2.ksc, s.ksc_ld);
            else s2.ksc = nullptr, s2.kmu = nullptr;
            // windowed precomputed K / V (QE_PATTN_KVPRE=0: off)
            if (pattn_kvpre() == 1 && fits && s.kvq && s.kvh && s.kvw > 0 && s.pf_p0 >= 0 && splits >= 2 && !s2.kmu) {
                int rows = 0;
                for (int j = 0; j < s.nkv; j++) rows = std::max(rows, pm_rank(s.kv_pm[j], s.pf_p0 + m));
                const int nwin = (rows + s.kvw - 1) / s.kvw;
                const int nb = ((m + 127) / 128) * s.nq * splits;
                for (int wi = 0; wi < nwin; wi++) {
                    pattn_kvpre_kernel<<<dim3((s.kvw + 7) / 8, s.nkv), 256, 0, st>>>((const uint8_t*)kc, (const uint8_t*)vc, s2, pos, m,
                                                                                   wi * s.kvw, s.kvq, s.kvh);
                    attn_pkv_kernel<<<nb, 256, kPKSmem, st>>>(q, s.kvq, s.kvh, s2, pos, m, work, wi * s.kvw, wi);
                }
                return;
            }
        }
        if (pf8_pp() && !qk8) {
            attn_pp8_kernel<<<((m + kPPRows - 1) / kPPRows) * s.nq * splits, kPPThreads, kPPSmem, st>>>(
                q, (const uint8_t*)kc, (const uint8_t*)vc, s2, proj, ld, pos, m, out, work);
        } else if (pf8_nw() == 4) {
            if (qk8) launch_pf8<true, 4>(m, splits, q, kc, vc, s2, proj, ld, pos, out, work, st);
            else launch_pf8<false, 4>(m, splits, q, kc, vc, s2, proj, ld, pos, out, work, st);
        } else {
            if (qk8) launch_pf8<true, 8>(m, splits, q, kc, vc, s2, proj, ld, pos, out, work, st);
            else launch_pf8<false, 8>(m, splits, q, kc, vc, s2, proj, ld, pos, out, work, st);
        }
        return;
    }
    const dim3 grid((m + 63) / 64, s.nq, splits);
    if (s.fp8) launch<4, true, false>(grid, q, kc, vc, s, proj, ld, pos, m, out, work, st);
    else launch<4, false, false>(grid, q, kc, vc, s, proj, ld, pos, m, out, work, st);
}

void attention_prefill_combine(const AttnShape& s, const float* proj, int ld, const int* pos, int m, bf16* out,
                               cudaStream_t st, float* work, int splits) {
    int nx = 0;  // exchange q heads
    for (int j = 0; j < s.nkv; j++)
        if (s.kv_x[j] == 3) nx += s.kv_qe[j] - s.kv_qb[j];
    const bool xchg = nx > 0;
    if (xchg) prefill_stats_kernel<<<(m * nx + 255) / 256, 256, 0, st>>>(work, splits, m, s, pos);
    if (splits > 1 || xchg)
        prefill_combine_kernel<<<dim3(m, s.nq), 256, 0, st>>>(work, splits, m, s, proj, ld, out, pos);
}

void attention_prefill(const bf16* q, const void* kc, const void* vc, const AttnShape& s,
                       const float* proj, int ld, const int* pos, int m, bf16* out,
                       cudaStream_t st, float* work, size_t work_floats) {
    bool xchg = false;
    for (int j = 0; j < s.nkv; j++) xchg |= s.kv_x[j] != 0;
    CHECK(!xchg || work, "position-split heads need the prefill work buffer");
    const int splits = attention_prefill_splits(s, m, work_floats, work != nullptr);
    attention_prefill_kernel(q, kc, vc, s, proj, ld, pos, m, out, st, work, splits);
    attention_prefill_combine(s, proj, ld, pos, m, out, st, work, splits);
}

static int dec8d_mode() {
    static const int d8d = env_int("QE_ATTN_DEC8D", 2);
    return d8d;
}
bool attention_dec8d_fused() { return dec8d_mode() == 2; }
// QE_D8_ROWSPLIT=1: one 16-row tile per attn_dec8d block, the tiles of a group in separate blocks
// (grid.y = splits x tiles): more blocks (warps) per SM for the same splits; K/V tiles are read once per row
// tile (the repeats from L2). Every row's arithmetic is unchanged: bit-identical.
static int d8_rowsplit() {  // 0 off, 1 always (microbench), 2 auto: one sequence, gated by context length
    static const int v = env_int("QE_D8_ROWSPLIT", 2);
    return v;
}
static bool has_split_heads(const AttnShape& s) {
    for (int j = 0; j < s.nkv; j++)
        if (s.kv_x[j] == 3) return true;
    return false;
}

void attention_dec8_seg(const AttnSegs& g, const AttnShape& s, int G, cudaStream_t st) {
    const int d8d = dec8d_mode();
    CHECK(d8d == 2 || !has_split_heads(s), "position-split heads need attn_dec8d (QE_ATTN_DEC8D=2)");
    const dim3 gd(s.nkv, kAttnSplits, g.n);
    const int rsm = d8_rowsplit();
    if (d8d == 2 && G > 16 && (rsm == 1 || (rsm == 2 && s.d8rs))) {
        const dim3 gr(s.nkv, kAttnSplits * ((G + 15) / 16), g.n);
        attn_dec8d_seg_kernel<1><<<gr, kD8Warps * 32, dec8d_smem<1>(), st>>>(g, s, 0);
        return;
    }
    if (d8d == 2 && G <= 16) attn_dec8d_seg_kernel<1><<<gd, kD8Warps * 32, dec8d_smem<1>(), st>>>(g, s);
    else if (d8d == 2 && G <= 32) attn_dec8d_seg_kernel<2><<<gd, kD8Warps * 32, dec8d_smem<2>(), st>>>(g, s);
    else if (G <= 16) launch_dec8_seg<1, kDecKG>(g, s, st);
    else if (G <= 32) launch_dec8_seg<2, kDecKG>(g, s, st);
    else if (d8d) attn_dec8d_seg_kernel<3><<<gd, kD8Warps * 32, dec8d_smem<3>(), st>>>(g, s);
    else launch_dec8_seg<3, kDecKG>(g, s, st);
}

bool attention_dec8_enabled() {
    static const bool on = env_set("QE_ATTN_DEC8") ? env_int("QE_ATTN_DEC8", true) != 0 : true;
    return on;
}

void attention_decode_tc(const bf16* q, const void* kc, const void* vc, const AttnShape& s,
                         const int* pos, int m, float* work, int G, cudaStream_t st) {
    const dim3 grid(s.nkv, kAttnSplits);
    if (s.fp8 && attention_dec8_enabled()) {
        // attn_dec8d for every group size (one summation order for any number of verified rows);
        // QE_ATTN_DEC8D=0 / 1: the older attn_dec8 for all / for G <= 32
        const int d8d = dec8d_mode();
        CHECK(d8d == 2 || !has_split_heads(s), "position-split heads need attn_dec8d (QE_ATTN_DEC8D=2)");
        const dim3 gd(s.nkv, kAttnSplits);
        const int rsm = d8_rowsplit();
        if (d8d == 2 && G > 16 && (rsm == 1 || (rsm == 2 && s.d8rs))) {
            attn_dec8d_kernel<1><<<dim3(s.nkv, kAttnSplits * ((G + 15) / 16)), kD8Warps * 32, dec8d_smem<1>(), st>>>(
                q, (const uint8_t*)kc, (const uint8_t*)vc, s, pos, m, work, 0);
            return;
        }
        if (d8d == 2 && G <= 16)
            attn_dec8d_kernel<1><<<gd, kD8Warps * 32, dec8d_smem<1>(), st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, s, pos, m, work);
        else if (d8d == 2 && G <= 32)
            attn_dec8d_kernel<2><<<gd, kD8Warps * 32, dec8d_smem<2>(), st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, s, pos, m, work);
        else if (G <= 16) launch_dec8<1, kDecKG>(q, kc, vc, s, pos, m, work, st);
        else if (G <= 32) launch_dec8<2, kDecKG>(q, kc, vc, s, pos, m, work, st);
        else if (d8d)
            attn_dec8d_kernel<3><<<gd, kD8Warps * 32, dec8d_smem<3>(), st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, s, pos, m, work);
        else launch_dec8<3, kDecKG>(q, kc, vc, s, pos, m, work, st);
        return;
    }
    if (s.fp8) {
        if (G <= 16) launch<1, true, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
        else if (G <= 32) launch<2, true, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
        else launch<3, true, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
    } else {
        if (G <= 16) launch<1, false, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
        else if (G <= 32) launch<2, false, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
        else launch<3, false, true>(grid, q, kc, vc, s, nullptr, 0, pos, m, nullptr, work, st);
    }
}

// ---------------------------------------------------------------- bench / precision helpers
namespace {
__device__ __forceinline__ unsigned pf_hash(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float pf_normal(size_t i, unsigned seed) {
    const unsigned a = pf_hash((unsigned)i * 2u + seed * 0x9e3779b9u), b = pf_hash((unsigned)(i >> 31) ^ a ^ 0x68e31da4u);
    const float u1 = (a >> 8) * (1.f / 16777216.f) + 1e-7f, u2 = (b >> 8) * (1.f / 16777216.f);
    return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
}
__global__ void pattn_fill_kernel(void* p, size_t n, int kind, float scale, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float x = pf_normal(i, seed) * scale;
        if (kind == 0) {  // e4m3 (|x| <= 448), round to nearest via fp32 -> e4m3 by hand
            float a = fminf(fabsf(x), 448.f);
            unsigned char b = 0;
            if (a >= 0.015625f) {  // normal: exponent e - 7
                int e;
                float f = frexpf(a, &e);         // a = f * 2^e, f in [0.5, 1)
                int mant = __float2int_rn((f * 2.f - 1.f) * 8.f);  // 3 bits
                int ex = e - 1 + 7;
                if (mant == 8) { mant = 0; ex++; }
                if (ex > 15 || (ex == 15 && mant == 7)) { ex = 15; mant = 6; }
                b = (unsigned char)((ex << 3) | mant);
            } else {  // subnormal: m * 2^-9
                b = (unsigned char)min(7, __float2int_rn(a * 512.f));
            }
            if (x < 0 && b) b |= 0x80;
            reinterpret_cast<unsigned char*>(p)[i] = b;
        } else {
            reinterpret_cast<bf16*>(p)[i] = __float2bfloat16(x);
        }
    }
}
__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 15, mm = b & 7;
    const float v = e ? ldexpf(1.f + mm / 8.f, e - 7) : ldexpf(mm / 8.f, -6);
    return s ? -v : v;
}
// block = (row, head), 256 threads: fp32 scores in shared memory chunks, two passes
__global__ void pattn_ref_kernel(const bf16* __restrict__ q, const unsigned char* __restrict__ kc,
                                 const unsigned char* __restrict__ vc, const AttnShape s, const int* __restrict__ pos,
                                 int r0, float* __restrict__ ref) {
    const int row = r0 + blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int kv = s.q2kv[h];
    const PosMap pm = s.kv_pm[kv];
    const int qp = pos[0] + row;
    const int L = pm_rank(pm, qp + 1);  // local keys with position <= qp
    __shared__ float qs[256];
    __shared__ float red[8];
    __shared__ float sc[1024];
    qs[d] = __bfloat162float(q[((size_t)row * s.nq + h) * 256 + d]);
    __syncthreads();
    const unsigned char* kb = kc + head_rows(s, kv, s.max_t) * 256;
    const unsigned char* vb = vc + head_rows(s, kv, s.max_t) * 256;
    const int warp = d >> 5, lane = d & 31;
    float mx = -FLT_MAX, den = 0.f, acc = 0.f;
    for (int c0 = 0; c0 < L; c0 += 1024) {
        const int n = min(1024, L - c0);
        for (int j = warp; j < n; j += 8) {  // warp per key
            const unsigned char* kr = kb + (size_t)(c0 + j) * 256;
            float a = 0.f;
            for (int e = lane; e < 256; e += 32) a += qs[e] * e4m3_to_float(kr[e]);
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) sc[j] = a * s.ks * 0.0625f;
        }
        __syncthreads();
        float cm = mx;
        for (int j = 0; j < n; j++) cm = fmaxf(cm, sc[j]);
        const float f = expf(mx - cm);
        acc *= f;
        den *= f;
        for (int j = 0; j < n; j++) {
            const float p = expf(sc[j] - cm);
            den += p;
            acc += p * e4m3_to_float(vb[(size_t)(c0 + j) * 256 + d]) * s.vs;
        }
        mx = cm;
        __syncthreads();
    }
    (void)red;
    ref[((size_t)blockIdx.x * s.nq + h) * 256 + d] = acc / den;
}
}  // namespace

void pattn_fill(void* p, size_t n, int kind, float scale, unsigned seed, cudaStream_t st) {
    pattn_fill_kernel<<<1024, 256, 0, st>>>(p, n, kind, scale, seed);
    CUDA_CHECK(cudaGetLastError());
}
void pattn_reference(const bf16* q, const void* kc, const void* vc, const AttnShape& s, const int* pos, int m,
                     int r0, int nr, float* ref, cudaStream_t st) {
    (void)m;
    pattn_ref_kernel<<<dim3(nr, s.nq), 256, 0, st>>>(q, (const unsigned char*)kc, (const unsigned char*)vc, s, pos, r0,
                                                     ref);
    CUDA_CHECK(cudaGetLastError());
}
