// Inkling 2 Mini prefill kernels ([ink-pf-*], branch inkling-prefill).
// Self-contained (only ink.h constants), so the kernel bench (tests/ink_pf_kbench.cu) builds it without the engine.
//
// Grouped expert GEMM (item 1). Slots = (token, k) routing entries sorted by expert (stable by entry index); an expert's
// slots are cut into units of <= 128. One block = 128 weight rows x one unit (128 slots), K in steps of 64, cp.async
// stages (weights as stored + the unit's activation rows gathered by index), mma.sync m16n8k16 f16 x f16 -> f32.
// Weights enter the MMA as exact fp16 values (no weight rounding):
//   NVFP4: e2m1 (bit placement: value * 2^-14) x (E4M3 scale * 2^7) = e2m1 * e4m3 * 2^-7: every product is a multiple
//          of 2^-17 and <= 21 in magnitude, so the fp16 product is exact; the epilogue multiplies by 2^7 * global.
//   MXFP8: e4m3 (bit placement: value * 2^-8) x 2^(s - 113) = e4m3 * 2^(s - 121) for E8M0 s in [113, 128] (checked at
//          load: the checkpoint has 113..118): multiples of 2^-17, <= 57,344: exact; the epilogue multiplies by 2^-6.
// Activations are rounded to fp16 (the only rounding before the fp32-accumulating MMA): x is the normed MLP input
// (|x| << 65,504), h = SwiGLU output (|h| <= 100); values below 2^-14 become fp16 subnormals (absolute error <= 2^-25).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "ink.h"

namespace ink {
namespace pf {

#ifndef XG_SCL2
#define XG_SCL2 1   // [ink-pf3-xscl]
#endif
#ifndef XG_SIMD4
#define XG_SIMD4 1
#endif
constexpr int XBM = 128, XBN = 128, XKS = 64;
constexpr int GQ_ = NQ / NKV;   // block: weight rows, slots, k per stage

__device__ __forceinline__ void cpa16(void* s, const void* g, int src) {   // src = 16 or 0 (zero fill)
    const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(g), "r"(src));
}
__device__ __forceinline__ void cpa4(void* s, const void* g) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    __half2 r = __hmul2(*(const __half2*)&a, *(const __half2*)&b);
    return *(const uint32_t*)&r;
}
// byte (low 8 bits of b): two e2m1 codes (low nibble = even element) -> half2 of value * 2^-14 (exact, 0.5 subnormal)
__device__ __forceinline__ uint32_t fp4h2(uint32_t b) {
    const uint32_t u = (b & 0xFu) | ((b & 0xF0u) << 12);
    return ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
}
// bytes 0 and 1 of b: two e4m3 values -> half2 of value * 2^-8 (exact, subnormals included)
__device__ __forceinline__ uint32_t fp8h2(uint32_t b) {
    const uint32_t u = __byte_perm(b, 0, 0x7170);   // byte 0 -> bits 0-7, byte 1 -> bits 16-23
    return ((u & 0x007f007fu) << 7) | ((u & 0x00800080u) << 8);
}
// NVFP4 block scale (E4M3, positive) as half2 of e4m3 * 2^7
__device__ __forceinline__ uint32_t nvs2(uint32_t s) {
    const uint32_t h = ((s & 0x7fu) << 7) | ((s & 0x80u) << 8);
    const __half v = __hmul(*(const __half*)&h, __float2half(32768.f));
    const __half2 r = __halves2half2(v, v);
    return *(const uint32_t*)&r;
}
// MXFP8 block scale (E8M0 in [113, 128]) as half2 of 2^(s - 113)
__device__ __forceinline__ uint32_t mxs2(uint32_t s) {
    const uint32_t h = (s - 98u) << 10;   // fp16 exponent field (s - 113) + 15
    return h | (h << 16);
}
__device__ __forceinline__ float pf_act(float g, float u) {   // == act_fn of the decode kernels (IEEE tanhf / expf)
    return 4.f * tanhf(g / 4.f) * (1.f / (1.f + expf(-g))) * 25.f * tanhf(u / 25.f);
}

template <int F_>
struct XgF {
    static constexpr int v = F_;
};
struct XgArgs {
    const uint8_t* cod;   // [E][N][row bytes]: NVFP4 K/2, MXFP8 K
    const uint8_t* scl;   // [E][N][K/16] E4M3 (NVFP4) / [E][N][K/32] E8M0 (MXFP8)
    const float* gsc;     // NVFP4 per-expert global scale (amax / denominator); MXFP8 null
    int N, K;
    const __half* X;      // activation rows (fp16), row stride K
    const int* pent;      // slot -> entry (token * 8 + k)
    const float* pw;      // slot -> routing weight
    const int* units;     // [u][3]: expert, first slot, count (<= 128)
    const int* nunits;
    int gather;           // 1: X row = pent[slot] >> 3 (gate/up reads token rows); 0: X row = slot (down reads h)
    __half* hout;         // EPI 0: [slot][N / 2] SwiGLU output
    float* yout;          // EPI 1: [entry][N] routing-weighted expert output
};
template <int FMT>
__host__ __device__ constexpr int xg_cb() { return FMT ? 64 : 32; }   // smem code row pitch (a 48 B pitch without the 2-way LDS conflicts: no gain)
template <int FMT, int BM = XBM>
__host__ __device__ constexpr int xg_stage_bytes() { return BM * xg_cb<FMT>() + (FMT ? 0 : BM * 4) + XBN * XKS * 2; }
template <int FMT, int NST, int BM = XBM>
__host__ __device__ constexpr int xg_smem(int K) { return NST * xg_stage_bytes<FMT, BM>(); }   // MXFP8 scales from registers

// grid (N / 128, max units); 256 threads = 8 warps: 4 along rows (32 each) x 2 along slots (64 each)
// BM weight rows per block ([ink-pf-xg-bm256]: 256 rows, 8 warps of 32 rows x 128 slots: half the ldmatrix per MMA,
// one block per SM, NST 3)
template <int FMT, int NST, int EPI, int WM = 4, int BM = XBM>   // WM warps along the rows (4: 32 rows x 64 slots per warp, 8: 16 x 128)
__global__ void __launch_bounds__(256, BM == XBM ? 2 : 1) k_xg(const __grid_constant__ XgArgs a) {
    constexpr int RW = BM / WM, MT = RW / 16, SW = XBN / (8 / WM), NTW = SW / 8;
    if ((int)blockIdx.y >= *a.nunits) return;
    extern __shared__ __align__(16) uint8_t xsm[];
    constexpr int CB = xg_cb<FMT>();                  // code row pitch in smem
    constexpr int SB = xg_stage_bytes<FMT, BM>();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp % WM, wn = warp / WM;
    const int u = blockIdx.y;
    const int e = a.units[3 * u], off = a.units[3 * u + 1], cnt = a.units[3 * u + 2];
    const int rb = blockIdx.x * BM;
    const int K = a.K, NK = K / XKS;
    const size_t rbytes = FMT ? (size_t)K : (size_t)K / 2, sbytes = FMT ? K / 32 : K / 16;
    const uint8_t* cbase = a.cod + ((size_t)e * a.N + rb) * rbytes;
    const uint8_t* sbase = a.scl + ((size_t)e * a.N + rb) * sbytes;
    // MXFP8 ([ink-pf-mx2]): the 2 E8M0 bytes of a stage per row read into registers one stage ahead (no 9 KB smem
    // row block: 2 blocks per SM)
    uint32_t mxc[MT][2] = {}, mxn[MT][2] = {};
    auto mxload = [&](int ks, uint32_t (&d)[MT][2]) {
        if (ks < NK)
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int h = 0; h < 2; h++) d[mt][h] = *(const uint16_t*)(sbase + (size_t)(wm * RW + mt * 16 + h * 8 + g) * sbytes + ks * 2);
    };
    // activation rows of this thread's copies: rows (tid >> 3) + 32 i, 16 B chunk tid & 7
    const int ch = tid & 7;
    const __half* xsrc[4];
    int xok[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = (tid >> 3) + 32 * i;
        xok[i] = r < cnt ? 16 : 0;
        const int row = r < cnt ? (a.gather ? a.pent[off + r] >> 3 : off + r) : 0;
        xsrc[i] = a.X + (size_t)row * K + ch * 8;
    }
    // [ink-pf-xg-epis] EPI 1: the unit's slot entries / routing weights staged in smem at the start (the epilogue read
    // them by __ldg per output pair: long-scoreboard stalls); visible after the first barrier of the k loop
    __shared__ int s_pent[EPI == 1 ? XBN : 1];
    __shared__ float s_pw[EPI == 1 ? XBN : 1];
    if (EPI == 1 && tid < XBN && tid < cnt) {
        s_pent[tid] = __ldg(a.pent + off + tid);
        s_pw[tid] = __ldg(a.pw + off + tid);
    }
    if (FMT) mxload(0, mxc);
    auto issue = [&](int ks) {
        if (ks < NK) {
            uint8_t* sw = xsm + (ks % NST) * SB;
            const int k0 = ks * XKS;
            if (FMT) {
#pragma unroll
                for (int i = 0; i < BM / 64; i++) {
                    const int q = tid + 256 * i, row = q >> 2, c = q & 3;
                    cpa16(sw + row * CB + c * 16, cbase + row * rbytes + k0 + c * 16, 16);
                }
            } else {
#pragma unroll
                for (int i = 0; i < BM / 128; i++) {
                    const int q = tid + 256 * i, row = q >> 1, c = q & 1;
                    cpa16(sw + row * CB + c * 16, cbase + row * rbytes + k0 / 2 + c * 16, 16);
                }
#pragma unroll
                for (int i = 0; i < (BM + 255) / 256; i++) {
                    const int r = tid + 256 * i;
                    if (r < BM) cpa4(sw + BM * CB + r * 4, sbase + r * sbytes + k0 / 16);
                }
            }
            uint8_t* sx = sw + BM * CB + (FMT ? 0 : BM * 4);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = (tid >> 3) + 32 * i;
                cpa16(sx + r * 128 + ((ch ^ (r & 7)) << 4), xsrc[i] + k0, xok[i]);
            }
        }
        cpa_commit();
    };
#pragma unroll
    for (int s = 0; s < NST - 1; s++) issue(s);
    float c[MT][NTW][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTW; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) c[i][j][q] = 0.f;
    const int nlive = cnt - wn * SW;   // slots of this warp (<= 0: the warp only copies)
    const int ntl = nlive >= SW ? NTW : nlive <= 0 ? 0 : (nlive + 7) >> 3;
    for (int ks = 0; ks < NK; ks++) {
        cpa_wait<NST - 2>();
        __syncthreads();
        issue(ks + NST - 1);
        if (FMT) {
            if (ks > 0) {
#pragma unroll
                for (int mt = 0; mt < MT; mt++) mxc[mt][0] = mxn[mt][0], mxc[mt][1] = mxn[mt][1];
            }
            mxload(ks + 1, mxn);
        }
        if (ntl == 0) continue;
        // [ink-pf-xg-full] the stage compiled per count of live n-tile pairs: no branch / WARPSYNC around every
        // ldmatrix + MMA pair
        auto stage = [&](auto FC) {
            constexpr int NPL = decltype(FC)::v;   // live n-tile pairs (compile time: no per-pair exit)
        const uint8_t* sw = xsm + (ks % NST) * SB;
        const uint8_t* sx = sw + BM * CB + (FMT ? 0 : BM * 4);
        uint32_t scw[MT][2];   // NVFP4: the 4 scale bytes of the stage per row (rows g, g + 8 of each m-tile)
        if (!FMT) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int h = 0; h < 2; h++) scw[mt][h] = *(const uint32_t*)(sw + BM * CB + (wm * RW + mt * 16 + h * 8 + g) * 4);
        }
#if XG_SCL2 && XG_SIMD4   // pf3_review L7: only the SIMD4 decode reads the pairs
        // [ink-pf3-xscl] the 4 scales per row as two half2 pairs (k16 steps 0-1, 2-3), each nvs2's value: e4m3 bits
        // -> fp16 * 2^-8 (fp8h2), * 2^15; per step one PRMT broadcast
        uint32_t scp[MT][2][2];
        if (!FMT) {
            const __half2 k15 = __float2half2_rn(32768.f);
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    scp[mt][h][0] = hmul2u(fp8h2(scw[mt][h]), *(const uint32_t*)&k15);
                    scp[mt][h][1] = hmul2u(fp8h2(scw[mt][h] >> 16), *(const uint32_t*)&k15);
                }
        }
#endif
#pragma unroll
        for (int j = 0; j < 4; j++) {   // k16 steps
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#if XG_SIMD4
                if (!FMT) {   // [ink-pf-xg-simd4] both rows' 4 code bytes in one word, nibbles -> fp16 high bytes 4 at a time
                    const int row = wm * RW + mt * 16 + g;
                    const uint2 wa = *(const uint2*)(sw + row * CB + j * 8), wb = *(const uint2*)(sw + (row + 8) * CB + j * 8);
                    const uint32_t sel = t | ((t + 4) << 4);
                    const uint32_t q = __byte_perm(__byte_perm(wa.x, wa.y, sel), __byte_perm(wb.x, wb.y, sel), 0x5410);   // [Ax Ay Bx By]
                    const uint32_t lo = ((q << 1) & 0x0E0E0E0Eu) | ((q << 4) & 0x80808080u);   // even codes: sign bit 7, magnitude bits 1-3
                    const uint32_t hi = ((q >> 3) & 0x0E0E0E0Eu) | (q & 0x80808080u);          // odd codes
                    const uint32_t u1 = __byte_perm(lo, hi, 0x5140), u2 = __byte_perm(lo, hi, 0x7362);   // [lo hi lo hi] of A / B
#if XG_SCL2 && XG_SIMD4
                    const uint32_t s2a = __byte_perm(scp[mt][0][j >> 1], 0u, (j & 1) ? 0x3232 : 0x1010);
                    const uint32_t s2b = __byte_perm(scp[mt][1][j >> 1], 0u, (j & 1) ? 0x3232 : 0x1010);
#else
                    const uint32_t s2a = nvs2((scw[mt][0] >> (8 * j)) & 255u), s2b = nvs2((scw[mt][1] >> (8 * j)) & 255u);
#endif
                    af[mt][0] = hmul2u(__byte_perm(u1, 0u, 0x1404), s2a);
                    af[mt][2] = hmul2u(__byte_perm(u1, 0u, 0x3424), s2a);
                    af[mt][1] = hmul2u(__byte_perm(u2, 0u, 0x1404), s2b);
                    af[mt][3] = hmul2u(__byte_perm(u2, 0u, 0x3424), s2b);
                } else
#endif
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int row = wm * RW + mt * 16 + h * 8 + g;
                    if (!FMT) {
                        const uint2 w = *(const uint2*)(sw + row * CB + j * 8);
                        const uint32_t s2 = nvs2((scw[mt][h] >> (8 * j)) & 255u);
                        af[mt][h] = hmul2u(fp4h2(w.x >> (8 * t)), s2);       // k 2t, 2t+1
                        af[mt][2 + h] = hmul2u(fp4h2(w.y >> (8 * t)), s2);   // k 2t+8, 2t+9
                    } else {
                        const uint4 w = *(const uint4*)(sw + row * CB + j * 16);
                        const uint32_t s2 = mxs2((mxc[mt][h] >> (8 * (j >> 1))) & 255u);
                        const uint32_t lo = (t & 2) ? w.y : w.x, hi = (t & 2) ? w.w : w.z;
                        af[mt][h] = hmul2u(fp8h2(lo >> (16 * (t & 1))), s2);
                        af[mt][2 + h] = hmul2u(fp8h2(hi >> (16 * (t & 1))), s2);
                    }
                }
#pragma unroll
            for (int p = 0; p < NTW / 2; p++) {
                if (p >= NPL) break;
                const int m = lane >> 3;
                const int n = wn * SW + (2 * p + (m >> 1)) * 8 + (lane & 7);
                const int cc = 2 * j + (m & 1);
                uint32_t b0, b1, b2, b3;
                ldm4(b0, b1, b2, b3, sx + n * 128 + ((cc ^ (n & 7)) << 4));
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    mma16(c[mt][2 * p], af[mt], b0, b1);
                    mma16(c[mt][2 * p + 1], af[mt], b2, b3);
                }
            }
        }
        };
        switch ((ntl + 1) >> 1) {
            case 1: stage(XgF<1>{}); break;
            case 2: stage(XgF<2>{}); break;
            case 3: stage(XgF<3>{}); break;
            case 4: stage(XgF<4>{}); break;
            case 5: stage(XgF<5 < NTW / 2 ? 5 : NTW / 2>{}); break;
            case 6: stage(XgF<6 < NTW / 2 ? 6 : NTW / 2>{}); break;
            case 7: stage(XgF<7 < NTW / 2 ? 7 : NTW / 2>{}); break;
            default: stage(XgF<NTW / 2>{}); break;
        }
    }
    cpa_wait<0>();
    if (ntl == 0) return;
    const float fe = FMT ? 0.015625f : 128.f * a.gsc[e];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int rA = rb + wm * RW + mt * 16 + g;   // rows rA, rA + 8
#pragma unroll
        for (int nt = 0; nt < NTW; nt++) {
            if (nt >= ntl) break;
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const int n = wn * SW + nt * 8 + 2 * t + q;
                const float va = c[mt][nt][q] * fe, vb = c[mt][nt][2 + q] * fe;
                if (EPI == 0) {   // rows 2i (gate) / 2i + 1 (up) sit in lane groups g / g + 1
                    // [ink-pf-xg-epi2] even lanes: output row rA / 2 (own gate va, partner's up va); odd lanes: row
                    // rA / 2 + 4 (partner's gate vb, own up vb): one shuffle, half the activations per lane
                    const bool ev = !(g & 1);
                    const float x = __shfl_xor_sync(0xffffffffu, ev ? vb : va, 4);
                    if (n < cnt) {
                        __half* ho = a.hout + (size_t)(off + n) * (a.N / 2) + (rA >> 1) + (ev ? 0 : 4);
                        *ho = __float2half_rn(ev ? pf_act(va, x) : pf_act(x, vb));
                    }
                } else if (n < cnt) {
                    const float w = s_pw[n];
                    float* yo = a.yout + (size_t)s_pent[n] * a.N + rA;
                    yo[0] = va * w;
                    yo[8] = vb * w;
                }
            }
        }
    }
}


// ---- prefill flash attention on tensor cores (item 4, [ink-pf-attn]). CTA = one KV head x 64 query rows x its 3 q
// heads (12 warps; warp = q head w / 4, 16 rows); keys in tiles of 32 staged through registers into a double buffer
// (bf16 cache -> fp16 smem). S = Q K^T (fp16 x fp16 -> f32), scale, relative bias G = r (16 x 16) x projT (16 x 48
// distances) by one mma per 8 distances (fp16 r / proj, f32 sums) through a per-warp smem row block, causal / window
// mask, online softmax in fp32 (IEEE expf), P fp16 -> O += P V (fp16). End: sink in the denominator, sigmoid(gate),
// bf16 AO. Keys come from the cache (ring slot = pos % cap); the chunk's K / V are written before this kernel.
struct A_pattn {
    const float* Q;      // [T][NQ][HD] (q post: clip, tau)
    const float* R;      // [T][NQ][DREL]
    const float* S;      // [T][NQ] sink logits
    const float* P;      // projection rows (gate at OFF_G), row stride ldp
    int ldp;
    const __half* projT;   // [REL][DREL] fp16
    const __nv_bfloat16* Kc;
    const __nv_bfloat16* Vc;
    int cap, ring, local;
    int pos, T;
    __nv_bfloat16* AO;   // [T][QD]
    float* AOf;          // optional fp32 copy (precision checks)
    const __nv_bfloat16* Ks;   // sliding layers: staging rows of the chunk (key pos + j at row j), null: all keys in the cache
    const __nv_bfloat16* Vs;
    int scap;
    // [ink-pf-qfuse] non-null: Q / R / S straight from P (k_qpost's arithmetic in the prologue; Q, R, S unused)
    const float* qs;
    const float* rs;
    int full;
    int kv0;   // [ink-tp-pf] a TP rank's first KV head (grid y = the rank's KV heads); 0 on one card
    // [ink-pf3-qkv] P in a TP rank's compact layout: column of segment X = OFF_X + head index - dX (0: full layout)
    int dq = 0, dr = 0, dg = 0, ds = 0;
};
#ifndef QF_UNR
#define QF_UNR 4
#endif
constexpr int QF_U = QF_UNR;   // [ink-pf-qfuse-u4] prologue rows unrolled (loads in flight)
// k_qpost's helpers (ink.cu), repeated here because this header is self-contained
__device__ __forceinline__ float pq_wsum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float pq_tau(int pos) { return 1.f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.f, 1.f)); }
constexpr float L2E = 1.4426950408889634f, QSC = 0.08838834764831845f * 1.4426950408889634f;   // log2 e, log2 e / sqrt(128)
__device__ __forceinline__ float ex2a(float x) {   // 2^x, MUFU (rel. error ~2^-22, far below the fp16 P rounding)
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));   // ftz: a 2^x below 2^-126 is 0 (P rounds to fp16 0 anyway; for l / cf / the sink term a flushed subnormal is below every fp32 sum it joins)
    return y;
}
constexpr int PA_KT = 32, PA_LD = HD + 8, PA_GW = 48;
#ifndef PA_PIPE
#define PA_PIPE 1
#endif
constexpr int PA_NB = PA_PIPE ? 3 : 2;   // K/V tile buffers ([ink-pf-attn-mbar]: 3, warps drift up to one tile)
constexpr int PA_SMEM = PA_NB * 2 * PA_KT * PA_LD * 2 + 12 * 16 * PA_GW * 4;
template <int M_>
struct PaM {
    static constexpr int v = M_;
};
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%0]; }" ::"r"((unsigned)__cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, int par) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(b);
    unsigned ok = 0;
    while (!ok)
        asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ uint32_t h2pack(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *(const uint32_t*)&v;
}
__device__ __forceinline__ uint32_t bf2h2(uint32_t u) {   // bf16 pair -> fp16 pair
    return h2pack(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ void ldm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
template <bool LOC, bool QF = false>   // [ink-pf-attn-tmpl] LOC: sliding layer (window, ring + staging rows); global layers compile without them; QF: [ink-pf-qfuse]
__global__ void __launch_bounds__(384, 1) k_pattn(const __grid_constant__ A_pattn a) {
    extern __shared__ __align__(16) uint8_t pasm[];
    __half* kvs = (__half*)pasm;   // [buf][K|V][PA_KT][PA_LD]
    float* Gs = (float*)(pasm + PA_NB * 2 * PA_KT * PA_LD * 2);
    const int qb = gridDim.x - 1 - blockIdx.x, kvh = a.kv0 + blockIdx.y;   // longest (last) query blocks first
    const int tid = threadIdx.x;
    __builtin_assume(tid < 384);
    const int w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hq = kvh * GQ_ + (w >> 2);
    const int rq = qb * 64 + (w & 3) * 16;   // warp's first row (chunk-relative)
    const int tA = rq + g, tB = tA + 8;
    const bool vA = tA < a.T, vB = tB < a.T;
    const int pA = a.pos + tA, pB = a.pos + tB;
    float* G = Gs + w * 16 * PA_GW;
    uint32_t qa[8][4];
    uint32_t ra[4];
    if constexpr (QF) {   // [ink-pf-qfuse] k_qpost per (row, head) in the warp: q = P q-part * qs, L2 clip (lane i + 32 k sums as
                  // its 4 warps), * tau; r likewise (16 lanes); Q / R never stored
        float fqA = 1.f, fqB = 1.f, frA = 1.f, frB = 1.f;
        const float* qsv = a.qs + hq * HD;
        const float qs0 = qsv[lane], qs1 = qsv[lane + 32], qs2 = qsv[lane + 64], qs3 = qsv[lane + 96];
        const float rsl = lane < DREL ? a.rs[hq * DREL + lane] : 0.f;
#pragma unroll QF_U
        for (int r = 0; r < 16; r++) {   // [ink-pf-qfuse-u4] 4 rows' loads in flight
            const float* pr = a.P + (size_t)min(rq + r, a.T - 1) * a.ldp;
            const float q0 = pr[OFF_Q - a.dq + hq * HD + lane] * qs0, q1 = pr[OFF_Q - a.dq + hq * HD + lane + 32] * qs1;
            const float q2 = pr[OFF_Q - a.dq + hq * HD + lane + 64] * qs2, q3 = pr[OFF_Q - a.dq + hq * HD + lane + 96] * qs3;
            const float s0 = pq_wsum(q0 * q0), s1 = pq_wsum(q1 * q1), s2 = pq_wsum(q2 * q2), s3 = pq_wsum(q3 * q3);
            const float nrm = sqrtf(s0 + s1 + s2 + s3);
            const float fq = fminf(QK_CLIP / (nrm + 1e-6f), 1.f);
            const float rv = lane < DREL ? pr[OFF_R - a.dr + hq * DREL + lane] * rsl : 0.f;
            const float rn = sqrtf(pq_wsum(rv * rv));
            const float fr = fminf(REL_CLIP / (rn + 1e-6f), 1.f);
            fqA = r == g ? fq : fqA;
            frA = r == g ? fr : frA;
            fqB = r == g + 8 ? fq : fqB;
            frB = r == g + 8 ? fr : frB;
        }
        const float tauA = a.full ? pq_tau(pA) : 1.f, tauB = a.full ? pq_tau(pB) : 1.f;
        const float* pa = a.P + (size_t)(vA ? tA : 0) * a.ldp;
        const float* pb = a.P + (size_t)(vB ? tB : 0) * a.ldp;
        auto qv = [&](const float* pp, float f, float tau, int c) { return ((pp[OFF_Q - a.dq + hq * HD + c] * qsv[c]) * f) * tau * QSC; };
        auto rv = [&](const float* pp, float f, float tau, int c) { return ((pp[OFF_R - a.dr + hq * DREL + c] * a.rs[hq * DREL + c]) * f) * tau * L2E; };
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const int c = 16 * s + 2 * t;
            qa[s][0] = vA ? h2pack(qv(pa, fqA, tauA, c), qv(pa, fqA, tauA, c + 1)) : 0u;
            qa[s][1] = vB ? h2pack(qv(pb, fqB, tauB, c), qv(pb, fqB, tauB, c + 1)) : 0u;
            qa[s][2] = vA ? h2pack(qv(pa, fqA, tauA, c + 8), qv(pa, fqA, tauA, c + 9)) : 0u;
            qa[s][3] = vB ? h2pack(qv(pb, fqB, tauB, c + 8), qv(pb, fqB, tauB, c + 9)) : 0u;
        }
        ra[0] = vA ? h2pack(rv(pa, frA, tauA, 2 * t), rv(pa, frA, tauA, 2 * t + 1)) : 0u;
        ra[1] = vB ? h2pack(rv(pb, frB, tauB, 2 * t), rv(pb, frB, tauB, 2 * t + 1)) : 0u;
        ra[2] = vA ? h2pack(rv(pa, frA, tauA, 8 + 2 * t), rv(pa, frA, tauA, 9 + 2 * t)) : 0u;
        ra[3] = vB ? h2pack(rv(pb, frB, tauB, 8 + 2 * t), rv(pb, frB, tauB, 9 + 2 * t)) : 0u;
    } else {
        {
            const float* qA = a.Q + ((size_t)(vA ? tA : 0) * NQ + hq) * HD;
            const float* qB = a.Q + ((size_t)(vB ? tB : 0) * NQ + hq) * HD;
#pragma unroll
            for (int s = 0; s < 8; s++) {
                const int c = 16 * s + 2 * t;
                qa[s][0] = vA ? h2pack(qA[c] * QSC, qA[c + 1] * QSC) : 0u;
                qa[s][1] = vB ? h2pack(qB[c] * QSC, qB[c + 1] * QSC) : 0u;
                qa[s][2] = vA ? h2pack(qA[c + 8] * QSC, qA[c + 9] * QSC) : 0u;
                qa[s][3] = vB ? h2pack(qB[c + 8] * QSC, qB[c + 9] * QSC) : 0u;
            }
        }
        const float* rA_ = a.R + ((size_t)(vA ? tA : 0) * NQ + hq) * DREL;
        const float* rB_ = a.R + ((size_t)(vB ? tB : 0) * NQ + hq) * DREL;
        ra[0] = vA ? h2pack(rA_[2 * t] * L2E, rA_[2 * t + 1] * L2E) : 0u;
        ra[1] = vB ? h2pack(rB_[2 * t] * L2E, rB_[2 * t + 1] * L2E) : 0u;
        ra[2] = vA ? h2pack(rA_[8 + 2 * t] * L2E, rA_[9 + 2 * t] * L2E) : 0u;
        ra[3] = vB ? h2pack(rB_[8 + 2 * t] * L2E, rB_[9 + 2 * t] * L2E) : 0u;
    }
    const int P0 = a.pos + qb * 64;
    const int qlast = min(a.pos + a.T - 1, P0 + 63);
    const int kbeg = LOC ? max(0, P0 - (WIN - 1)) : 0;
    const int ntile = (qlast - kbeg) / PA_KT + 1;
    const int wq0 = a.pos + rq, wq1 = min(wq0 + 15, a.pos + a.T - 1);   // the warp's live query positions
    // staging: 1,024 16 B chunks per tile (K then V, 32 keys x 16 chunks), 3 per thread
    uint4 stg[3];
    // [ink-pf-attn-ld] straight-line loads: every thread issues its 3 chunk loads (chunk index clamped to 1,023: the
    // third chunk of threads >= 256 re-reads a valid chunk and is not stored), source picked by selects, the ring slot
    // by a mask (cap is a power of two when ring): no divergent blocks, so the 3 loads are not serialised by the
    // scoreboard waits of the branchy form
    const int cmask = a.cap - 1;
    const int kvrow = kvh * a.cap;
    auto gload = [&](int it) {
        const int K0 = kbeg + it * PA_KT;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int i = min(tid + 384 * j, 1023);
            const int key = min(K0 + ((i & 511) >> 4), qlast), ch = i & 15;
            const __nv_bfloat16* src;
            if (LOC) {
                const bool stg_row = a.Ks != nullptr && key >= a.pos;
                const __nv_bfloat16* cbase = i < 512 ? a.Kc : a.Vc;
                const __nv_bfloat16* sbase = i < 512 ? a.Ks : a.Vs;
                const int ci = (kvrow + (a.ring ? (key & cmask) : key)) * HD + ch * 8;
                const int si = (kvh * a.scap + (key - a.pos)) * HD + ch * 8;
                src = stg_row ? sbase + si : cbase + ci;
            } else
                src = (i < 512 ? a.Kc : a.Vc) + (size_t)((kvrow + key) * HD + ch * 8);
            stg[j] = __ldg((const uint4*)src);
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int i = tid + 384 * j;
            if (i < 1024) {
                const int kv = i >> 9, key = (i & 511) >> 4, ch = i & 15;
                uint4 v = stg[j];
                v.x = bf2h2(v.x);
                v.y = bf2h2(v.y);
                v.z = bf2h2(v.z);
                v.w = bf2h2(v.w);
                *(uint4*)(kvs + ((buf * 2 + kv) * PA_KT + key) * PA_LD + ch * 8) = v;
            }
        }
    };
#if PA_PIPE
    // [ink-pf-attn-mbar] 3 tile buffers with per-buffer full / empty mbarriers (one arrival per warp) instead of one
    // __syncthreads per tile: a warp may run up to one tile ahead of the slowest one, so the 3 warps of a scheduler
    // stop entering the softmax (no tensor work) in lockstep
    __shared__ uint64_t fullb[3], emptyb[3];
    if (tid == 0)
        for (int b = 0; b < 3; b++) mb_init(&fullb[b], 12), mb_init(&emptyb[b], 12);
    __syncthreads();
    gload(0);
    sstore(0);
    if (ntile > 1) {
        gload(1);
        sstore(1);
    }
    __syncthreads();
#else
    gload(0);
    sstore(0);
    __syncthreads();
#endif
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
#if PA_PIPE
    int buf = 0, nb = 2, fph = 0, eph = 0;   // tile it in buffer buf; per-buffer phase bits of full[] / empty[]
#endif
    for (int it = 0; it < ntile; it++) {
#if PA_PIPE
        const int K0 = kbeg + it * PA_KT;
        if (it + 2 < ntile) gload(it + 2);
        if (it >= 2) {   // tiles >= 2 arrive on full[]
            mb_wait(&fullb[buf], (fph >> buf) & 1);
            fph ^= 1 << buf;
        }
#else
        const int buf = it & 1, K0 = kbeg + it * PA_KT;
        if (it + 1 < ntile) gload(it + 1);
#endif
        const bool live = wq0 <= wq1 && K0 <= wq1 && (!LOC || K0 + PA_KT - 1 >= wq0 - (WIN - 1));
        // [ink-pf-attn-spec] the tile body compiled for 3 cases (warp-uniform choice): M 2 = no mask, no bias (interior
        // tiles of global layers), 1 = no mask (bias computed), 0 = general
        auto tile = [&](auto MC) {
            constexpr int M = decltype(MC)::v;
            const __half* ks = kvs + (buf * 2 + 0) * PA_KT * PA_LD;
            const __half* vs = kvs + (buf * 2 + 1) * PA_KT * PA_LD;
            float sc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; s++)
#pragma unroll
                for (int pp = 0; pp < 2; pp++) {
                    const int mm = lane >> 3;
                    const int key = (2 * pp + (mm >> 1)) * 8 + (lane & 7), cc = 2 * s + (mm & 1);
                    uint32_t b0, b1, b2, b3;
                    ldm4(b0, b1, b2, b3, ks + key * PA_LD + cc * 8);
                    mma16(sc[2 * pp], qa[s], b0, b1);
                    mma16(sc[2 * pp + 1], qa[s], b2, b3);
                }
            // relative bias: distances d = q - k, warp range [wq0 - K0 - 31, wq0 + 15 - K0]
            const int dlo = wq0 - K0 - (PA_KT - 1);
            const bool bias = M != 2 && dlo < REL;
            if (bias) {
#pragma unroll
                for (int jt = 0; jt < PA_GW / 8; jt++) {
                    const int d = dlo + jt * 8 + g;
                    uint32_t b0 = 0u, b1 = 0u;
                    if (d >= 0 && d < REL) {
                        b0 = *(const uint32_t*)(a.projT + d * DREL + 2 * t);
                        b1 = *(const uint32_t*)(a.projT + d * DREL + 8 + 2 * t);
                    }
                    float gc[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16(gc, ra, b0, b1);
                    G[g * PA_GW + jt * 8 + 2 * t] = gc[0];
                    G[g * PA_GW + jt * 8 + 2 * t + 1] = gc[1];
                    G[(g + 8) * PA_GW + jt * 8 + 2 * t] = gc[2];
                    G[(g + 8) * PA_GW + jt * 8 + 2 * t + 1] = gc[3];
                }
                __syncwarp();
            }
            {
            float mx[2] = {-INFINITY, -INFINITY};
            // no mask: every row valid, every key <= the first row, inside the last row's window
            const bool nomask = M != 0 || rq + 15 < a.T && K0 + PA_KT - 1 <= wq0 && (!LOC || K0 >= wq0 + 15 - (WIN - 1));
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = (e >> 1) ? g + 8 : g, c = nt * 8 + 2 * t + (e & 1);
                    float v = sc[nt][e];
                    if (bias) v += G[r * PA_GW + r - c + (PA_KT - 1)];
                    if (!nomask) {
                        const int qp = (e >> 1) ? pB : pA, key = K0 + c;
                        const bool ok = ((e >> 1) ? vB : vA) && key <= qp && (!LOC || key >= qp - (WIN - 1));
                        v = ok ? v : -INFINITY;
                    }
                    sc[nt][e] = v;
                    mx[e >> 1] = fmaxf(mx[e >> 1], v);
                }
            if (bias) __syncwarp();
#pragma unroll
            for (int h = 0; h < 2; h++) {
                mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
                mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
            }
            float cf[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const float mn = fmaxf(m[h], mx[h]);
                cf[h] = ex2a(m[h] - mn);   // m = -inf: 0 (mn finite) / NaN guarded below
                if (mn == -INFINITY) cf[h] = 1.f;
                m[h] = mn;
            }
            float ps[2] = {0.f, 0.f};
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float mr = m[e >> 1] == -INFINITY ? 0.f : m[e >> 1];
                    const float pv = ex2a(sc[nt][e] - mr);
                    sc[nt][e] = pv;
                    ps[e >> 1] += pv;
                }
            l[0] = l[0] * cf[0] + ps[0];
            l[1] = l[1] * cf[1] + ps[1];
            // [ink-pf-attn-rskip] unmasked tiles: skip the O rescale when no row max of the warp moved (cf == 1 exactly:
            // multiplying by 1 is the identity, bit-exact)
            if (M == 0 || __any_sync(0xffffffffu, cf[0] != 1.f || cf[1] != 1.f))
#pragma unroll
            for (int j = 0; j < 16; j++) {
                o[j][0] *= cf[0];
                o[j][1] *= cf[0];
                o[j][2] *= cf[1];
                o[j][3] *= cf[1];
            }
            }
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                uint32_t pa[4];
                pa[0] = h2pack(sc[2 * kk][0], sc[2 * kk][1]);
                pa[1] = h2pack(sc[2 * kk][2], sc[2 * kk][3]);
                pa[2] = h2pack(sc[2 * kk + 1][0], sc[2 * kk + 1][1]);
                pa[3] = h2pack(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
#pragma unroll
                for (int nd = 0; nd < 16; nd += 2) {
                    const int mm = lane >> 3;
                    const int key = 16 * kk + (mm & 1) * 8 + (lane & 7), dim = (nd + (mm >> 1)) * 8;
                    uint32_t b0, b1, b2, b3;
                    ldm4t(b0, b1, b2, b3, vs + key * PA_LD + dim);
                    mma16(o[nd], pa, b0, b1);
                    mma16(o[nd + 1], pa, b2, b3);
                }
            }
                };
        if (live) {
            const bool nm = rq + 15 < a.T && K0 + PA_KT - 1 <= wq0 && (!LOC || K0 >= wq0 + 15 - (WIN - 1));
            if (nm && wq0 - K0 - (PA_KT - 1) >= REL) tile(PaM<2>{});
            else if (nm) tile(PaM<1>{});
            else tile(PaM<0>{});
        }
#if PA_PIPE
        __syncwarp();   // review2 L7: lane 0 arrives for the whole warp (barrier count 12 = warps): the warp's reads / stores first
        if (lane == 0) mb_arrive(&emptyb[buf]);
        if (it + 2 < ntile) {   // buffer nb last held tile it - 1 (it >= 1)
            if (it >= 1) {
                mb_wait(&emptyb[nb], (eph >> nb) & 1);
                eph ^= 1 << nb;
            }
            sstore(nb);
            __syncwarp();
            if (lane == 0) mb_arrive(&fullb[nb]);
        }
        buf = buf == 2 ? 0 : buf + 1;
        nb = nb == 2 ? 0 : nb + 1;
#else
        if (it + 1 < ntile) {
            sstore(buf ^ 1);   // the other buffer was last read in iteration it - 1 (before the barrier below it)
        }
        __syncthreads();
#endif
    }
    // finish: quad sums of l, sink, gate
#pragma unroll
    for (int h = 0; h < 2; h++) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const bool v = h ? vB : vA;
        if (!v) continue;
        const int tt = h ? tB : tA;
        const float sink = (QF ? a.P[(size_t)tt * a.ldp + OFF_S - a.ds + hq] : a.S[(size_t)tt * NQ + hq]) * L2E;   // log2 units like the scores
        const float M = fmaxf(m[h], sink);
        const float sc_ = m[h] == -INFINITY ? 0.f : ex2a(m[h] - M);
        const float L = l[h] * sc_ + ex2a(sink - M);
        const float f = sc_ / L;
        const float* gp = a.P + (size_t)tt * a.ldp + OFF_G - a.dg + hq * HD;
        __nv_bfloat16* ao = a.AO + (size_t)tt * QD + hq * HD;
#pragma unroll
        for (int nd = 0; nd < 16; nd++) {
            const int d = nd * 8 + 2 * t;
            const float o0 = o[nd][2 * h] * f, o1 = o[nd][2 * h + 1] * f;
            const float g0 = gp[d], g1 = gp[d + 1];
            const float y0 = o0 * (1.f / (1.f + expf(-g0))), y1 = o1 * (1.f / (1.f + expf(-g1)));
            *(__nv_bfloat162*)(ao + d) = __floats2bfloat162_rn(y0, y1);
            if (a.AOf) {
                a.AOf[(size_t)tt * QD + hq * HD + d] = y0;
                a.AOf[(size_t)tt * QD + hq * HD + d + 1] = y1;
            }
        }
    }
}

// staging rows j0 + blockIdx.x -> ring slot (pos + j) % cap; grid (rows, NKV), 32 threads x 16 B (K) + 16 B (V)
__global__ void k_ringcp(const __nv_bfloat16* Ks, const __nv_bfloat16* Vs, int scap, int j0, __nv_bfloat16* K, __nv_bfloat16* V, int cap, int pos,
                         int h0 = 0) {
    const int j = j0 + blockIdx.x, h = h0 + blockIdx.y, c = threadIdx.x & 15;
    const size_t src = ((size_t)h * scap + j) * HD + c * 8, dst = ((size_t)h * cap + (pos + j) % cap) * HD + c * 8;
    if (threadIdx.x < 16) *(uint4*)(K + dst) = *(const uint4*)(Ks + src);
    else *(uint4*)(V + dst) = *(const uint4*)(Vs + src);
}
// fp32 reference merge of k_attn_part partials (as k_attn_comb, fp32 out): precision checks only
__global__ void k_pcomb_ref(const float* pm, const float* pl, const float* pa, int NS, const float* S, const float* P, int ldp, float* AOf) {
    const int t = blockIdx.x, h = blockIdx.y, dd = threadIdx.x;
    const size_t b = ((size_t)t * NQ + h) * NS;
    const float sink = S[(size_t)t * NQ + h];
    float M = sink;
    for (int z = 0; z < NS; z++) M = fmaxf(M, pm[b + z]);
    float L = expf(sink - M), o = 0.f;
    for (int z = 0; z < NS; z++)
        if (pm[b + z] != -INFINITY) {
            const float e = expf(pm[b + z] - M);
            L += pl[b + z] * e;
            o += pa[(b + z) * HD + dd] * e;
        }
    o /= L;
    const float g = P[(size_t)t * ldp + OFF_G + h * HD + dd];
    AOf[(size_t)t * QD + h * HD + dd] = o * (1.f / (1.f + expf(-g)));
}
__global__ void k_errsum(const float* a, const float* b, size_t n, double* acc) {   // acc[0] += (a-b)^2, acc[1] += b^2, acc[2] = max |a-b|
    double e = 0, r = 0;
    float mx = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const double d = (double)a[i] - b[i];
        e += d * d;
        r += (double)b[i] * b[i];
        mx = fmaxf(mx, fabsf(a[i] - b[i]));
    }
    atomicAdd(acc, e);
    atomicAdd(acc + 1, r);
    atomicMax((unsigned*)(acc + 2), __float_as_uint(mx));
}


// ---- message gates for T rows on tensor cores ([ink-pf-mg]). fp32 activations enter the bf16 MMA as hi + lo bf16
// pairs (x = hi + lo, |lo| <= 2^-9 |x|; two MMAs per tile: ~16-bit operands, f32 sums), weights exact bf16.
//   k_pmgz: z[g][t][r] = rsqrt(ms(h)) (Wd_h h)_r + rsqrt(ms(m)) (Wd_m m)_r; grid (ceil(T / 64), ng), 4 warps x 16 rows
//   k_pmgo: out = (1 - tanh(Wf z)) h + tanh(Wg z) m; grid (ceil(T / 64), H / 64, ng), 8 warps (16 rows x 32 channels)
struct PMg {
    const float* h[3];
    const float* m[3];
    const __nv_bfloat16* down[3];
    const __nv_bfloat16* f[3];
    const __nv_bfloat16* g[3];
    float* out[3];
    float* z;
    int T;
    float* zp;    // K-split partials [ks][g][T][2 GR] (h | m dots), null: one pass
    float* ssp;   // [ks][g][T][2] sums of squares
};
__device__ __forceinline__ void hilo(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    const float2 hf = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(a - hf.x, b - hf.y);
    hi = *(const uint32_t*)&h;
    lo = *(const uint32_t*)&l;
}
__device__ __forceinline__ void mmab(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
constexpr int MG_KC = 64;   // k columns per staged weight chunk
__global__ void __launch_bounds__(128) k_pmgz(const __grid_constant__ PMg a) {
    // [ink-pf-mgz-cp] weight chunks double-buffered by cp.async (as k_prouter); same sums in the same order
    __shared__ __align__(16) __nv_bfloat16 wsb[2][2][GR][MG_KC + 8];   // [buf][part h|m][row][k]
    const int gi = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t0 = blockIdx.x * 64 + w * 16;
    const int tA = min(t0 + g, a.T - 1), tB = min(t0 + g + 8, a.T - 1);
    const float* hA = a.h[gi] + (size_t)tA * H;
    const float* hB = a.h[gi] + (size_t)tB * H;
    const float* mA = a.m[gi] + (size_t)tA * H;
    const float* mB = a.m[gi] + (size_t)tB * H;
    const __nv_bfloat16* W = a.down[gi];
    float ch[8][4], cm[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) ch[i][j] = cm[i][j] = 0.f;
    float ss[4] = {0.f, 0.f, 0.f, 0.f};   // h rows A / B, m rows A / B
    const int KS = gridDim.z, kz = blockIdx.z;   // K split ([ink-pf-mgz-ks]): small chunks fill the SMs
    const int k0 = kz * (H / KS), k1 = (kz + 1) * (H / KS);
    auto wload = [&](int kc, int b) {
#pragma unroll
        for (int i = 0; i < 2 * GR * (MG_KC / 8) / 128; i++) {   // 16 B pieces
            const int q = tid + 128 * i, part = q / (GR * (MG_KC / 8)), r = (q / (MG_KC / 8)) % GR, c8 = q % (MG_KC / 8);
            cpa16(&wsb[b][part][r][c8 * 8], W + (size_t)r * 2 * H + part * H + kc + c8 * 8, 16);
        }
        cpa_commit();
    };
    wload(k0, 0);
    for (int kc = k0, bb = 0; kc < k1; kc += MG_KC, bb ^= 1) {
        if (kc + MG_KC < k1) {
            wload(kc + MG_KC, bb ^ 1);   // buffer bb ^ 1 was last read in the previous chunk (barrier at its end)
            cpa_wait<1>();
        } else
            cpa_wait<0>();
        __syncthreads();
        __nv_bfloat16 (*ws)[GR][MG_KC + 8] = wsb[bb];
#pragma unroll
        for (int s = 0; s < MG_KC / 16; s++) {
            const int c = kc + 16 * s + 2 * t;
            uint32_t ah[4], al[4], mh[4], ml[4];
            {
                const float2 x0 = *(const float2*)(hA + c), x1 = *(const float2*)(hB + c), x2 = *(const float2*)(hA + c + 8),
                             x3 = *(const float2*)(hB + c + 8);
                ss[0] += x0.x * x0.x + x0.y * x0.y + x2.x * x2.x + x2.y * x2.y;
                ss[1] += x1.x * x1.x + x1.y * x1.y + x3.x * x3.x + x3.y * x3.y;
                hilo(x0.x, x0.y, ah[0], al[0]);
                hilo(x1.x, x1.y, ah[1], al[1]);
                hilo(x2.x, x2.y, ah[2], al[2]);
                hilo(x3.x, x3.y, ah[3], al[3]);
            }
            {
                const float2 x0 = *(const float2*)(mA + c), x1 = *(const float2*)(mB + c), x2 = *(const float2*)(mA + c + 8),
                             x3 = *(const float2*)(mB + c + 8);
                ss[2] += x0.x * x0.x + x0.y * x0.y + x2.x * x2.x + x2.y * x2.y;
                ss[3] += x1.x * x1.x + x1.y * x1.y + x3.x * x3.x + x3.y * x3.y;
                hilo(x0.x, x0.y, mh[0], ml[0]);
                hilo(x1.x, x1.y, mh[1], ml[1]);
                hilo(x2.x, x2.y, mh[2], ml[2]);
                hilo(x3.x, x3.y, mh[3], ml[3]);
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int r = nt * 8 + g, kk = 16 * s + 2 * t;
                const uint32_t b0 = *(const uint32_t*)&ws[0][r][kk], b1 = *(const uint32_t*)&ws[0][r][kk + 8];
                const uint32_t d0 = *(const uint32_t*)&ws[1][r][kk], d1 = *(const uint32_t*)&ws[1][r][kk + 8];
                mmab(ch[nt], ah, b0, b1);
                mmab(ch[nt], al, b0, b1);
                mmab(cm[nt], mh, d0, d1);
                mmab(cm[nt], ml, d0, d1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ss[i] += __shfl_xor_sync(0xffffffffu, ss[i], 1);
        ss[i] += __shfl_xor_sync(0xffffffffu, ss[i], 2);
    }
    if (a.zp) {   // partials; k_pmgzr sums them in split order
        const int rA = t0 + g, rB = rA + 8;
        float* zp = a.zp + ((size_t)kz * gridDim.y + gi) * (size_t)a.T * 2 * GR;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int r = nt * 8 + 2 * t + e;
                if (rA < a.T) {
                    zp[(size_t)rA * 2 * GR + r] = ch[nt][e];
                    zp[(size_t)rA * 2 * GR + GR + r] = cm[nt][e];
                }
                if (rB < a.T) {
                    zp[(size_t)rB * 2 * GR + r] = ch[nt][2 + e];
                    zp[(size_t)rB * 2 * GR + GR + r] = cm[nt][2 + e];
                }
            }
        if (t == 0) {
            float* sp = a.ssp + ((size_t)kz * gridDim.y + gi) * (size_t)a.T * 2;
            if (rA < a.T) { sp[2 * rA] = ss[0]; sp[2 * rA + 1] = ss[2]; }
            if (rB < a.T) { sp[2 * rB] = ss[1]; sp[2 * rB + 1] = ss[3]; }
        }
        return;
    }
    const float ihA = rsqrtf(ss[0] / H + 1e-6f), ihB = rsqrtf(ss[1] / H + 1e-6f), imA = rsqrtf(ss[2] / H + 1e-6f),
                imB = rsqrtf(ss[3] / H + 1e-6f);
    const int rA = t0 + g, rB = rA + 8;
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int r = nt * 8 + 2 * t + e;
            if (rA < a.T) a.z[((size_t)gi * a.T + rA) * GR + r] = ch[nt][e] * ihA + cm[nt][e] * imA;
            if (rB < a.T) a.z[((size_t)gi * a.T + rB) * GR + r] = ch[nt][2 + e] * ihB + cm[nt][2 + e] * imB;
        }
}
// K-split reduce: z[g][t][r] = (sum_ks h-dot) rsqrt(ms h) + (sum_ks m-dot) rsqrt(ms m), splits summed in order
__global__ void k_pmgzr(const float* zp, const float* ssp, int KS, int ng, int T, float* z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng * T * GR) return;
    const int r = i % GR, tg = i / GR;   // tg = gi * T + t
    float zh = 0.f, zm = 0.f, sh = 0.f, sm = 0.f;
    for (int k = 0; k < KS; k++) {
        const size_t b = (size_t)k * ng * T + tg;
        zh += zp[b * 2 * GR + r];
        zm += zp[b * 2 * GR + GR + r];
        sh += ssp[b * 2];
        sm += ssp[b * 2 + 1];
    }
    z[(size_t)tg * GR + r] = zh * rsqrtf(sh / H + 1e-6f) + zm * rsqrtf(sm / H + 1e-6f);
}
__global__ void __launch_bounds__(256, 2) k_pmgo(const __grid_constant__ PMg a) {
    const int gi = blockIdx.z, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t0 = blockIdx.x * 64 + (w & 3) * 16, c0 = blockIdx.y * 64 + (w >> 2) * 32;
    const int tA = min(t0 + g, a.T - 1), tB = min(t0 + g + 8, a.T - 1);
    // [ink-pf-mgo-smem] the block's z rows (64 tokens) and f / g rows (64 channels) staged once by cp.async: one load
    // latency instead of one per k step; same values, same MMA order
    __shared__ __align__(16) float zs[64][GR + 8];
    __shared__ __align__(16) __nv_bfloat16 fs[64][GR + 8], gs[64][GR + 8];
    {
        const int tb = blockIdx.x * 64, cb = blockIdx.y * 64;
#pragma unroll
        for (int i = 0; i < 64 * GR / 4 / 256; i++) {   // z: 16 B = 4 floats
            const int q = tid + 256 * i, r = q / (GR / 4), c4 = q % (GR / 4);
            cpa16(&zs[r][c4 * 4], a.z + ((size_t)gi * a.T + min(tb + r, a.T - 1)) * GR + c4 * 4, 16);
        }
#pragma unroll
        for (int i = 0; i < 64 * GR / 8 / 256; i++) {   // f / g: 16 B = 8 bf16
            const int q = tid + 256 * i, r = q / (GR / 8), c8 = q % (GR / 8);
            cpa16(&fs[r][c8 * 8], a.f[gi] + (size_t)(cb + r) * GR + c8 * 8, 16);
            cpa16(&gs[r][c8 * 8], a.g[gi] + (size_t)(cb + r) * GR + c8 * 8, 16);
        }
        cpa_commit();
    }
    const float* zA = &zs[(w & 3) * 16 + g][0];
    const float* zB = &zs[(w & 3) * 16 + g + 8][0];
    float cf[4][4], cg[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) cf[i][j] = cg[i][j] = 0.f;
    const int rA = t0 + g, rB = rA + 8;
    float2 hv[4][2], mv[4][2];   // the epilogue's h / m loads issued first (in flight under the MMAs)
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const size_t i = (size_t)(hh ? tB : tA) * H + c0 + nt * 8 + 2 * t;
            hv[nt][hh] = *(const float2*)(a.h[gi] + i);
            mv[nt][hh] = *(const float2*)(a.m[gi] + i);
        }
    cpa_wait<0>();
    __syncthreads();
#pragma unroll
    for (int s = 0; s < GR / 16; s++) {
        const int c = 16 * s + 2 * t;
        uint32_t zh[4], zl[4];
        {
            const float2 x0 = *(const float2*)(zA + c), x1 = *(const float2*)(zB + c), x2 = *(const float2*)(zA + c + 8),
                         x3 = *(const float2*)(zB + c + 8);
            hilo(x0.x, x0.y, zh[0], zl[0]);
            hilo(x1.x, x1.y, zh[1], zl[1]);
            hilo(x2.x, x2.y, zh[2], zl[2]);
            hilo(x3.x, x3.y, zh[3], zl[3]);
        }
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int rl = c0 - blockIdx.y * 64 + nt * 8 + g;
            const uint32_t f0 = *(const uint32_t*)&fs[rl][c], f1 = *(const uint32_t*)&fs[rl][c + 8];
            const uint32_t g0 = *(const uint32_t*)&gs[rl][c], g1 = *(const uint32_t*)&gs[rl][c + 8];
            mmab(cf[nt], zh, f0, f1);
            mmab(cf[nt], zl, f0, f1);
            mmab(cg[nt], zh, g0, g1);
            mmab(cg[nt], zl, g0, g1);
        }
    }
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {   // channel pairs: one 8 B load / store per tensor
            const int r = hh ? rB : rA, ch = c0 + nt * 8 + 2 * t;
            if (r >= a.T) continue;
            const size_t i = (size_t)r * H + ch;
            float2 o;
            o.x = (1.f - tanhf(cf[nt][2 * hh])) * hv[nt][hh].x + tanhf(cg[nt][2 * hh]) * mv[nt][hh].x;
            o.y = (1.f - tanhf(cf[nt][2 * hh + 1])) * hv[nt][hh].y + tanhf(cg[nt][2 * hh + 1]) * mv[nt][hh].y;
            *(float2*)(a.out[gi] + i) = o;
        }
}


// router scores of T rows on tensor cores ([ink-pf-router]): sc[t][e] = sigmoid(Wr[e] . x[t]), x fp32 as bf16 hi + lo
// (as k_pmgz), Wr exact bf16, f32 sums. grid (ceil(T / (16 PR_W)), NE / PR_NEB), PR_W warps x 16 rows x PR_NEB experts.
#ifndef PR_NEB_
#define PR_NEB_ 64   // nsys 2K: 128 per block 125 us, 32: 116, 64: 86.5
#endif
constexpr int PR_NEB = PR_NEB_;
constexpr int PR_W = 2;   // warps (16 rows each) per router block: 32-row blocks fill more SMs (same per-row sum order)
// [ink-pf-router-ne] NEB experts per block (grid.y = NE / NEB): more blocks on the SMs; each expert's dot keeps its sum
// order (bit-identical); block row 0 of grid.y writes the fp16 copy
template <int NEB = NE>
__global__ void __launch_bounds__(32 * PR_W) k_prouter(const float* X, const __nv_bfloat16* Wr, int T, float* sc, __half* X16 = nullptr) {
    constexpr int NTE = NEB / 8;
    static_assert(NE % NEB == 0 && NEB % 8 == 0 && (NEB * (MG_KC / 8)) % (32 * PR_W) == 0, "PR_NEB");   // review3 L3
    const int e0 = blockIdx.y * NEB;
    if (blockIdx.y) X16 = nullptr;
    Wr += (size_t)e0 * H;
    // [ink-pf-router-cp] weight chunks double-buffered by cp.async (the next chunk in flight under the MMAs)
    __shared__ __align__(16) __nv_bfloat16 wsb[2][NEB][MG_KC + 8];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t0 = blockIdx.x * (16 * PR_W) + w * 16;
    const float* xA = X + (size_t)min(t0 + g, T - 1) * H;
    const float* xB = X + (size_t)min(t0 + g + 8, T - 1) * H;
    float c[NTE][4];
#pragma unroll
    for (int i = 0; i < NTE; i++) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    auto wload = [&](int kc, int b) {
#pragma unroll
        for (int i = 0; i < NEB * (MG_KC / 8) / (32 * PR_W); i++) {
            const int q = tid + 32 * PR_W * i, r = q / (MG_KC / 8), c8 = q % (MG_KC / 8);
            cpa16(&wsb[b][r][c8 * 8], Wr + (size_t)r * H + kc + c8 * 8, 16);
        }
        cpa_commit();
    };
    wload(0, 0);
    for (int kc = 0, b = 0; kc < H; kc += MG_KC, b ^= 1) {
        if (kc + MG_KC < H) {
            wload(kc + MG_KC, b ^ 1);   // buffer b ^ 1 was last read in the previous chunk (barrier below it)
            cpa_wait<1>();
        } else
            cpa_wait<0>();
        __syncthreads();
        __nv_bfloat16 (*ws)[MG_KC + 8] = wsb[b];
#pragma unroll
        for (int s = 0; s < MG_KC / 16; s++) {
            const int cc = kc + 16 * s + 2 * t;
            uint32_t ah[4], al[4];
            const float2 x0 = *(const float2*)(xA + cc), x1 = *(const float2*)(xB + cc), x2 = *(const float2*)(xA + cc + 8),
                         x3 = *(const float2*)(xB + cc + 8);
            hilo(x0.x, x0.y, ah[0], al[0]);
            hilo(x1.x, x1.y, ah[1], al[1]);
            hilo(x2.x, x2.y, ah[2], al[2]);
            hilo(x3.x, x3.y, ah[3], al[3]);
            if (X16) {   // the fp16 copy for the expert GEMM (each element of the block's rows loaded exactly once here)
                if (t0 + g < T) {
                    *(__half2*)(X16 + (size_t)(t0 + g) * H + cc) = __floats2half2_rn(x0.x, x0.y);
                    *(__half2*)(X16 + (size_t)(t0 + g) * H + cc + 8) = __floats2half2_rn(x2.x, x2.y);
                }
                if (t0 + g + 8 < T) {
                    *(__half2*)(X16 + (size_t)(t0 + g + 8) * H + cc) = __floats2half2_rn(x1.x, x1.y);
                    *(__half2*)(X16 + (size_t)(t0 + g + 8) * H + cc + 8) = __floats2half2_rn(x3.x, x3.y);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTE; nt++) {
                const int r = nt * 8 + g, kk = 16 * s + 2 * t;
                const uint32_t b0 = *(const uint32_t*)&ws[r][kk], b1 = *(const uint32_t*)&ws[r][kk + 8];
                mmab(c[nt], ah, b0, b1);
                mmab(c[nt], al, b0, b1);
            }
        }
        __syncthreads();
    }
    const int rA = t0 + g, rB = rA + 8;
#pragma unroll
    for (int nt = 0; nt < NTE; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int ex = e0 + nt * 8 + 2 * t + e;
            if (rA < T) sc[(size_t)rA * NE + ex] = 1.f / (1.f + expf(-c[nt][e]));
            if (rB < T) sc[(size_t)rB * NE + ex] = 1.f / (1.f + expf(-c[nt][2 + e]));
        }
}


// combine + post-MoE conv fused ([ink-pf-rowfuse]): MO(t) = sum over k (order 0..7) of Y[t * 8 + k] (as k_pcomb),
// out[t] = add[t] + (w3 MO(t) + w2 MO(t-1) + w1 MO(t-2) + w0 MO(t-3) + MO(t)) / sqrt(5) (as k_conv; MO(t < 0) from the
// conv state). A thread owns one channel over a run of PC_R tokens (the 3 rows before the run recomputed); the last 3
// rows' MO go to MOs (the state update after the kernel). Bit-identical to k_pcomb + k_conv.
constexpr int PC_R = 64;   // [ink-pf-pconv64] 3 recomputed rows per run: 9.4 % -> 4.7 % extra reads
__global__ void __launch_bounds__(128) k_pconv(const float* Y, int T, const float* st, const float* w, const float* add, float* out, float* MOs) {
    const int c = blockIdx.x * 128 + threadIdx.x, t0 = blockIdx.y * PC_R, t1 = min(T, t0 + PC_R);
    auto mo = [&](int t) -> float {
        if (t < 0) return st[(3 + t) * H + c];
        const float* y = Y + (size_t)t * TOPK * H + c;
        float s = y[0];
#pragma unroll
        for (int k = 1; k < TOPK; k++) s += y[(size_t)k * H];
        return s;
    };
    float x1 = mo(t0 - 1), x2 = mo(t0 - 2), x3 = mo(t0 - 3);
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = t0; t < t1; t++) {
        const float x0 = mo(t);
        const float o = __fmaf_rn(w0, x3, __fmaf_rn(w1, x2, __fmaf_rn(w3, x0, __fmul_rn(w2, x1))));   // k_conv's SASS order
        const size_t i = (size_t)t * H + c;
        out[i] = __fmaf_rn(__fadd_rn(o, x0), 1.f / sqrtf(5.f), add ? add[i] : 0.f);
        if (t >= T - 3) MOs[i] = x0;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

__global__ void k_addpos(const int* p, int d, int* out) { *out = *p + d; }
// ---- routing (item 2): top-8 per token (the decode path's selection function: same order, weights and NaN guard; the
// prefill scores come from the hi/lo TC router, ~16-bit x: near-ties may differ from decode, judged by the KL gate),
// expert counts + unit plan, stable placement by entry index. Deterministic: no atomics on the placement path.
// one warp per token; identical arithmetic to block_topk (ink.cu)
__global__ void __launch_bounds__(256) k_ptopk(const float* sc, const float* bias, int T, int* ids, float* wts) {
    const int tk = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (tk >= T) return;
    float s[4], v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        s[j] = sc[(size_t)tk * NE + lane + 32 * j];
        v[j] = s[j] + bias[lane + 32 * j];
    }
    float sum = 0.f, mine = 0.f;
    int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (v[j] > bv) {
                bv = v[j];
                bi = lane + 32 * j;
            }
        for (int o = 16; o; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
            if (v2 > bv || (v2 == bv && i2 < bi)) {
                bv = v2;
                bi = i2;
            }
        }
        if (bi >= NE) bi = k;
        const int bj = (bi >> 5) & 3;
        const float sl = bj == 0 ? s[0] : bj == 1 ? s[1] : bj == 2 ? s[2] : s[3];
        const float sb = __shfl_sync(0xffffffffu, sl, bi & 31);
        sum += sb;
        if (lane == k) {
            mine = sb;
            myid = bi;
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (bi == lane + 32 * j) v[j] = -INFINITY;
    }
    const float w = mine / (sum + 1e-20f) * ROUTE_SCALE;
    if (lane < TOPK) {
        ids[tk * TOPK + lane] = myid;
        wts[tk * TOPK + lane] = w;
    }
}
// one block of 1024: counts, offsets off[0..128], units of <= 128 slots per expert; nunits
__global__ void __launch_bounds__(1024) k_pplan(const int* ids, int n8, int* off, int* units, int* nunits) {
    __shared__ int hist[NE], uo[NE + 1];
    const int t = threadIdx.x;
    if (t < NE) hist[t] = 0;
    __syncthreads();
    for (int i = t; i < n8; i += 1024) atomicAdd(&hist[ids[i]], 1);
    __syncthreads();
    if (t == 0) {
        int o = 0, q = 0;
        for (int x = 0; x < NE; x++) {
            off[x] = o;
            uo[x] = q;
            o += hist[x];
            q += (hist[x] + XBN - 1) / XBN;
        }
        off[NE] = o;
        uo[NE] = q;
        *nunits = q;
    }
    __syncthreads();
    if (t < NE) {
        const int n = hist[t], o = off[t];
        for (int i = 0; i * XBN < n; i++) {
            int* p = units + 3 * (uo[t] + i);
            p[0] = t;
            p[1] = o + i * XBN;
            p[2] = min(XBN, n - i * XBN);
        }
    }
}
// block = expert: its entries in entry order -> slots off[e] ..
// [ink-pf-pfill2] coalesced tiles of PF_NT entries, warp ballots + a block prefix (same slots: entry order); the old form
// scanned 64-entry segments per thread and summed the counts serially
constexpr int PF_NT = 256;   // threads (entries per tile; 1024: 33.7 us vs 32.5 at 2K)
__global__ void __launch_bounds__(PF_NT) k_pfill(const int* ids, const float* wts, int n8, const int* off, int* pent, float* pw) {
    __shared__ int wc[PF_NT / 32];
    const int e = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
    int o = off[e];
    if (off[e + 1] == o) return;
    for (int base = 0; base < n8; base += PF_NT) {
        const int i = base + t;
        const bool hit = i < n8 && ids[i] == e;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) wc[w] = __popc(m);
        __syncthreads();
        int before = 0, tot = 0;
#pragma unroll
        for (int j = 0; j < PF_NT / 32; j++) {
            before += j < w ? wc[j] : 0;
            tot += wc[j];
        }
        if (hit) {
            const int s = o + before + __popc(m & ((1u << lane) - 1u));
            pent[s] = i;
            pw[s] = wts[i];
        }
        o += tot;
        __syncthreads();   // wc reuse
    }
}
__global__ void k_f2h(const float* x, __half* y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
// MO[t][c] = sum over k (fixed order 0..7) of Y[t * 8 + k][c]
__global__ void k_pcomb(const float* Y, int T, float* MO) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * (H / 4)) return;
    const size_t tk = i / (H / 4), c4 = i % (H / 4);
    float4 s = ((const float4*)(Y + tk * TOPK * H))[c4];
#pragma unroll
    for (int k = 1; k < TOPK; k++) {
        const float4 v = ((const float4*)(Y + (tk * TOPK + k) * H))[c4];
        s.x += v.x;
        s.y += v.y;
        s.z += v.z;
        s.w += v.w;
    }
    ((float4*)(MO + tk * H))[c4] = s;
}

}  // namespace pf
}  // namespace ink
