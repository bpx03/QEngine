#pragma once
// [dsv4-pf2-cpu] prefill routed experts of few rows on the CPU (host copy in RAM, engine layout of gguf_iq.h).
// Weights: the GPU's exact bf16 values bf16_rn(sgn(db * g) * d) (every fp32 product is exact, so g * (d * db) is the
// same number); x = the FP8-rounded bf16 rows the GPU uses (xq); vdpbf16ps with fp32 accumulation (order differs from
// the GPU's). GLU (clamps, routing weight) and q_fp8 of the hidden as k_glu_rows + k_q_fp8 (same expressions; CPU expf
// may differ from CUDA expf by an ulp: review cpu L1). Included by dsv4_cpu.cpp
// (built with -march=znver4: AVX512_BF16 / VBMI / BITALG).
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include "gguf_iq.h"

namespace dsv4 {
namespace pfc {
namespace {   // [dsv4-rel3-isa] internal linkage: the AVX-512 copies never become shared (weak) symbols

alignas(64) inline float GF2XXS[256][8], GF3XXS[256][4], GF2S[1024][8], GF3S[512][4];
inline void tables_init() {
    static bool done = false;
    if (done) return;
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 8; j++) GF2XXS[i][j] = (float)((gg_iq2xxs_grid[i] >> (8 * j)) & 255);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 4; j++) GF3XXS[i][j] = (float)((gg_iq3xxs_grid[i] >> (8 * j)) & 255);
    for (int i = 0; i < 1024; i++)
        for (int j = 0; j < 8; j++) GF2S[i][j] = (float)((gg_iq2s_grid[i] >> (8 * j)) & 255);
    for (int i = 0; i < 512; i++)
        for (int j = 0; j < 4; j++) GF3S[i][j] = (float)((gg_iq3s_grid[i] >> (8 * j)) & 255);
    done = true;
}
// 8 qwords -> per qword the 4 sign bytes of 7-bit fields at bit offsets o, o+7, o+14, o+21 (even parity in bit 7)
inline __m256i signs8(__m512i v, long long sel) {
    __m512i f = _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_set1_epi64(sel), v), _mm512_set1_epi8(127));
    const __m512i par = _mm512_and_si512(_mm512_popcnt_epi8(f), _mm512_set1_epi8(1));
    f = _mm512_or_si512(f, _mm512_slli_epi16(par, 7));
    return _mm512_cvtepi64_epi32(f);
}
// per format: prep (one 256-weight block: sign masks sm[8] of 32 bits, scales e[8][2] = d * db of each half)
// and the 32 weights of sub-block k as bf16
template <int MF> struct Blk;
template <> struct Blk<XF_IQ2XXS> {
    static constexpr int CB = 64;
    uint32_t sm[8]; float e[8];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        const __m512i v = _mm512_loadu_si512(c);   // qword k = grid indices | aux << 32
        _mm256_storeu_si256((__m256i*)sm, signs8(v, 0x00000000352e2720ll));
        const __m256 s = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(_mm512_srli_epi64(v, 60)));
        _mm256_storeu_ps(e, _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(0.5f), s), _mm256_set1_ps(0.25f)), _mm256_set1_ps(d)));
    }
    inline void w(int k, __m512& lo, __m512& hi, __mmask16& ml, __mmask16& mh, float& el, float& eh) const {
        uint32_t vx;
        memcpy(&vx, c + 8 * k, 4);
        lo = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(GF2XXS[vx & 255])), _mm256_load_ps(GF2XXS[(vx >> 8) & 255]), 1);
        hi = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(GF2XXS[(vx >> 16) & 255])), _mm256_load_ps(GF2XXS[vx >> 24]), 1);
        ml = (__mmask16)(sm[k] & 0xffff); mh = (__mmask16)(sm[k] >> 16); el = eh = e[k];
    }
};
template <> struct Blk<XF_IQ3XXS> {
    static constexpr int CB = 96;
    uint32_t sm[8]; float e[8];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        const __m512i v = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)(c + 64)));
        _mm256_storeu_si256((__m256i*)sm, signs8(v, 0x00000000150e0700ll));
        const __m256 s = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(_mm512_srli_epi64(v, 28)));
        _mm256_storeu_ps(e, _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(0.5f), s), _mm256_set1_ps(0.5f)), _mm256_set1_ps(d)));
    }
    static inline __m512 q4(const float (*G)[4], const uint8_t* ix) {
        __m512 r = _mm512_castps128_ps512(_mm_load_ps(G[ix[0]]));
        r = _mm512_insertf32x4(r, _mm_load_ps(G[ix[1]]), 1);
        r = _mm512_insertf32x4(r, _mm_load_ps(G[ix[2]]), 2);
        return _mm512_insertf32x4(r, _mm_load_ps(G[ix[3]]), 3);
    }
    inline void w(int k, __m512& lo, __m512& hi, __mmask16& ml, __mmask16& mh, float& el, float& eh) const {
        lo = q4(GF3XXS, c + 8 * k);
        hi = q4(GF3XXS, c + 8 * k + 4);
        ml = (__mmask16)(sm[k] & 0xffff); mh = (__mmask16)(sm[k] >> 16); el = eh = e[k];
    }
};
template <> struct Blk<XF_IQ2S> {
    static constexpr int CB = 80;
    float e[8][2];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        for (int k = 0; k < 8; k++) {
            e[k][0] = d * ((0.5f + (float)(c[72 + k] & 15)) * 0.25f);
            e[k][1] = d * ((0.5f + (float)(c[72 + k] >> 4)) * 0.25f);
        }
    }
    inline void w(int k, __m512& lo, __m512& hi, __mmask16& ml, __mmask16& mh, float& el, float& eh) const {
        const uint8_t* qs = c + 4 * k;
        const uint32_t h = c[64 + k];
        lo = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(GF2S[qs[0] | ((h & 3) << 8)])),
                                _mm256_load_ps(GF2S[qs[1] | (((h >> 2) & 3) << 8)]), 1);
        hi = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(GF2S[qs[2] | (((h >> 4) & 3) << 8)])),
                                _mm256_load_ps(GF2S[qs[3] | (((h >> 6) & 3) << 8)]), 1);
        uint32_t s;
        memcpy(&s, c + 32 + 4 * k, 4);
        ml = (__mmask16)(s & 0xffff); mh = (__mmask16)(s >> 16); el = e[k][0]; eh = e[k][1];
    }
};
template <> struct Blk<XF_IQ3S> {
    static constexpr int CB = 108;
    float e[8];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        for (int k = 0; k < 8; k++) e[k] = d * (float)(1 + 2 * ((c[104 + k / 2] >> (4 * (k & 1))) & 15));
    }
    inline void w(int k, __m512& lo, __m512& hi, __mmask16& ml, __mmask16& mh, float& el, float& eh) const {
        const uint8_t* qs = c + 8 * k;
        const uint32_t h = c[64 + k];
        __m512 r = _mm512_castps128_ps512(_mm_load_ps(GF3S[qs[0] | ((h & 1) << 8)]));
        r = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[1] | (((h >> 1) & 1) << 8)]), 1);
        r = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[2] | (((h >> 2) & 1) << 8)]), 2);
        lo = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[3] | (((h >> 3) & 1) << 8)]), 3);
        r = _mm512_castps128_ps512(_mm_load_ps(GF3S[qs[4] | (((h >> 4) & 1) << 8)]));
        r = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[5] | (((h >> 5) & 1) << 8)]), 1);
        r = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[6] | (((h >> 6) & 1) << 8)]), 2);
        hi = _mm512_insertf32x4(r, _mm_load_ps(GF3S[qs[7] | (((h >> 7) & 1) << 8)]), 3);
        uint32_t s;
        memcpy(&s, c + 72 + 4 * k, 4);
        ml = (__mmask16)(s & 0xffff); mh = (__mmask16)(s >> 16); el = eh = e[k];
    }
};
// y[r * ldy + row] = sum_k W[row][k] x[r][k] for rows [0, N) of one matrix (codes q, fp16 d ds), x [M][K] bf16
template <int MF, int M, int R>
inline void mat(const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
    const int KB = K / 256;
    const __m512 sb = _mm512_castsi512_ps(_mm512_set1_epi32((int)0x80000000));
    for (int row = 0; row < N; row += R) {
        __m512 acc[R][M];
        for (int i = 0; i < R; i++)
            for (int r = 0; r < M; r++) acc[i][r] = _mm512_setzero_ps();
        for (int b = 0; b < KB; b++) {
            Blk<MF> B[R];
            for (int i = 0; i < R; i++)
                B[i].prep(q + ((size_t)(row + i) * KB + b) * Blk<MF>::CB, _cvtsh_ss(ds[(size_t)(row + i) * KB + b]));
            for (int k = 0; k < 8; k++) {
                __m512bh w[R];
                for (int i = 0; i < R; i++) {
                    __m512 lo, hi;
                    __mmask16 ml, mh;
                    float el, eh;
                    B[i].w(k, lo, hi, ml, mh, el, eh);
                    lo = _mm512_mul_ps(lo, _mm512_set1_ps(el));
                    hi = _mm512_mul_ps(hi, _mm512_set1_ps(eh));
                    lo = _mm512_mask_xor_ps(lo, ml, lo, sb);
                    hi = _mm512_mask_xor_ps(hi, mh, hi, sb);
                    w[i] = _mm512_cvtne2ps_pbh(hi, lo);
                }
                const int ko = b * 256 + 32 * k;
                for (int r = 0; r < M; r++) {
                    const __m512bh xv = (__m512bh)_mm512_loadu_si512(x + (size_t)r * K + ko);
                    for (int i = 0; i < R; i++) acc[i][r] = _mm512_dpbf16_ps(acc[i][r], w[i], xv);
                }
            }
        }
        for (int i = 0; i < R; i++)
            for (int r = 0; r < M; r++) y[(size_t)r * ldy + row + i] = _mm512_reduce_add_ps(acc[i][r]);
    }
}
template <int MF, int M>
inline void mat_m(const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
    if constexpr (M <= 4) mat<MF, M, 4>(q, ds, N, K, x, y, ldy);
    else mat<MF, M, 2>(q, ds, N, K, x, y, ldy);
}
template <int MF>
inline void mat_any(int m, const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
    switch (m) {
        case 1: mat_m<MF, 1>(q, ds, N, K, x, y, ldy); break;
        case 2: mat_m<MF, 2>(q, ds, N, K, x, y, ldy); break;
        case 3: mat_m<MF, 3>(q, ds, N, K, x, y, ldy); break;
        case 4: mat_m<MF, 4>(q, ds, N, K, x, y, ldy); break;
        case 5: mat_m<MF, 5>(q, ds, N, K, x, y, ldy); break;
        case 6: mat_m<MF, 6>(q, ds, N, K, x, y, ldy); break;
        case 7: mat_m<MF, 7>(q, ds, N, K, x, y, ldy); break;
        default: mat_m<MF, 8>(q, ds, N, K, x, y, ldy); break;
    }
}
// one matrix of format mf (an i-quant matrix format), m <= 8 rows
inline bool mat_fmt_ok(int mf) { return mf == XF_IQ2XXS || mf == XF_IQ3XXS || mf == XF_IQ2S || mf == XF_IQ3S; }
inline void mat_run(int mf, int m, const uint8_t* q, const void* s, int N, int K, const uint16_t* x, float* y, int ldy) {
    const uint16_t* ds = (const uint16_t*)s;
    if (mf == XF_IQ2XXS) mat_any<XF_IQ2XXS>(m, q, ds, N, K, x, y, ldy);
    else if (mf == XF_IQ3XXS) mat_any<XF_IQ3XXS>(m, q, ds, N, K, x, y, ldy);
    else if (mf == XF_IQ2S) mat_any<XF_IQ2S>(m, q, ds, N, K, x, y, ldy);
    else mat_any<XF_IQ3S>(m, q, ds, N, K, x, y, ldy);
}
// GPU-equal helpers (dsv4_kernels.cuh k_glu_rows / k_q_fp8)
inline float e4m3_rne(float v) {
    const float a = fabsf(v);
    if (a == 0.f) return v;
    int ex;
    frexpf(a, &ex);
    const float qn = ldexpf(1.f, std::max(ex - 1, -6) - 3);
    return copysignf(nearbyintf(a / qn) * qn, v);
}
inline float pow2ceil(float v) {
    int ex;
    const float mm = frexpf(v, &ex);
    if (mm == 0.5f) ex -= 1;
    return ldexpf(1.f, ex);
}
inline uint16_t bf16_bits(float f) {   // exact for FP8 values (3-bit mantissa): truncation = rounding
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
// one routed expert of m rows: xg [m][D] gathered bf16 rows, wt[m] routing weights, Y [m][D] fp32 (ldy D);
// scratch: h13 [m][2 INTER] floats, hq [m][INTER] uint16
inline void expert(XFmt f, const uint8_t* base, int m, const uint16_t* xg, const float* wt, float* h13, uint16_t* hq, float* Y) {
    MatView mv[3];
    expert_views(f, base, mv);
    mat_run(mat_fmt(f, 0), m, mv[0].q, mv[0].s, INTER, D, xg, h13, 2 * INTER);
    mat_run(mat_fmt(f, 1), m, mv[1].q, mv[1].s, INTER, D, xg, h13 + INTER, 2 * INTER);
    for (int i = 0; i < m; i++) {
        const float* H = h13 + (size_t)i * 2 * INTER;
        alignas(64) float h[INTER];
        for (int c = 0; c < INTER; c++) {
            const float g = fminf(H[c], CLAMP);
            const float u = fminf(fmaxf(H[INTER + c], -CLAMP), CLAMP);
            h[c] = wt[i] * ((g / (1.f + expf(-g))) * u);
        }
        for (int b = 0; b < INTER; b += 32) {
            float am = 0.f;
            for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(h[b + j]));
            am = fmaxf(am, 1e-4f);
            const float s = pow2ceil(am * (1.f / 448.f));
            for (int j = 0; j < 32; j++)
                hq[(size_t)i * INTER + b + j] = bf16_bits(e4m3_rne(fminf(fmaxf(h[b + j] / s, -448.f), 448.f)) * s);
        }
    }
    mat_run(mat_fmt(f, 2), m, mv[2].q, mv[2].s, D, INTER, hq, Y, D);
}

}  // namespace
}  // namespace pfc
}  // namespace dsv4
