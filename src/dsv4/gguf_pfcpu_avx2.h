#pragma once
// [dsv4-rel3-isa] AVX2 (x86-64-v3) build of gguf_pfcpu.h (prefill routed experts of few rows on the CPU), namespace
// pfc2 (the AVX-512 build's pfc functions are inline: another namespace keeps the two builds apart at link time).
// Same numbers as the AVX-512 build: the weights are the same fp32 products (scale, sign) rounded to bf16 (RNE, as
// vcvtne2ps2bf16; exact here, the products are bf16 values), and each fp32 lane j of the AVX-512 accumulator takes
// vdpbf16ps's two products in its order: acc + w[2j+1] x[2j+1], then + w[2j] x[2j] (each product exact in fp32, so an
// FMA = the instruction's add); the 16 lanes are two ymm halves, reduced in _mm512_reduce_add_ps's order.
// Unlike the BF16 instructions, this keeps denormals (they flush them): weights >= grid x d x db and FP8-rounded x keep
// every product and sum far above 1e-38 here, so the bits agree (review rel3 L3).
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include "gguf_iq.h"

namespace dsv4 {
namespace pfc2 {

alignas(64) static float GF2XXS[256][8], GF3XXS[256][4], GF2S[1024][8], GF3S[512][4];
static inline void tables_init() {
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
static inline uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint32_t signs4(uint32_t aux) {   // fields at bits 0 / 7 / 14 / 21 -> 4 sign bytes with parity
    return (uint32_t)gg_ksign(aux & 127) | ((uint32_t)gg_ksign((aux >> 7) & 127) << 8) |
           ((uint32_t)gg_ksign((aux >> 14) & 127) << 16) | ((uint32_t)gg_ksign((aux >> 21) & 127) << 24);
}
// the 32 weights of sub-block k: w4[0..3] = weights 0-7, 8-15, 16-23, 24-31 (grid values), sm = sign bits, el / eh
// = scale of weights 0-15 / 16-31
template <int MF> struct Blk;
template <> struct Blk<XF_IQ2XXS> {
    static constexpr int CB = 64;
    uint32_t sm[8]; float e[8];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        for (int k = 0; k < 8; k++) {
            const uint32_t aux = ld32(c + 8 * k + 4);
            sm[k] = signs4(aux);
            e[k] = ((0.5f + (float)(int)(aux >> 28)) * 0.25f) * d;
        }
    }
    inline void w(int k, __m256* w4, uint32_t& s, float& el, float& eh) const {
        const uint8_t* q = c + 8 * k;
        for (int i = 0; i < 4; i++) w4[i] = _mm256_load_ps(GF2XXS[q[i]]);
        s = sm[k]; el = eh = e[k];
    }
};
template <> struct Blk<XF_IQ3XXS> {
    static constexpr int CB = 96;
    uint32_t sm[8]; float e[8];
    const uint8_t* c;
    inline void prep(const uint8_t* cc, float d) {
        c = cc;
        for (int k = 0; k < 8; k++) {
            const uint32_t aux = ld32(c + 64 + 4 * k);
            sm[k] = signs4(aux);
            e[k] = ((0.5f + (float)(int)(aux >> 28)) * 0.5f) * d;
        }
    }
    inline void w(int k, __m256* w4, uint32_t& s, float& el, float& eh) const {
        const uint8_t* ix = c + 8 * k;
        for (int i = 0; i < 4; i++)
            w4[i] = _mm256_set_m128(_mm_load_ps(GF3XXS[ix[2 * i + 1]]), _mm_load_ps(GF3XXS[ix[2 * i]]));
        s = sm[k]; el = eh = e[k];
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
    inline void w(int k, __m256* w4, uint32_t& s, float& el, float& eh) const {
        const uint8_t* qs = c + 4 * k;
        const uint32_t h = c[64 + k];
        for (int i = 0; i < 4; i++) w4[i] = _mm256_load_ps(GF2S[qs[i] | (((h >> (2 * i)) & 3) << 8)]);
        s = ld32(c + 32 + 4 * k); el = e[k][0]; eh = e[k][1];
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
    inline void w(int k, __m256* w4, uint32_t& s, float& el, float& eh) const {
        const uint8_t* qs = c + 8 * k;
        const uint32_t h = c[64 + k];
        for (int i = 0; i < 4; i++)
            w4[i] = _mm256_set_m128(_mm_load_ps(GF3S[qs[2 * i + 1] | (((h >> (2 * i + 1)) & 1) << 8)]),
                                    _mm_load_ps(GF3S[qs[2 * i] | (((h >> (2 * i)) & 1) << 8)]));
        s = ld32(c + 72 + 4 * k); el = eh = e[k];
    }
};
// 8 sign bits -> sign-bit dwords
static inline __m256 sgn8(uint32_t b) {
    const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i m = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32((int)b), bit), bit);
    return _mm256_castsi256_ps(_mm256_and_si256(m, _mm256_set1_epi32((int)0x80000000)));
}
// fp32 -> the bf16 value (round to nearest even, as vcvtne2ps2bf16 for finite values) kept as fp32
static inline __m256 bf16r(__m256 v) {
    const __m256i u = _mm256_castps_si256(v);
    const __m256i r = _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1)));
    return _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32((int)0xffff0000)));
}
// even / odd elements of 16 floats (a = 0..7, b = 8..15) -> 8 lanes each
static inline void deint(__m256 a, __m256 b, __m256& ev, __m256& od) {
    ev = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
    od = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
}
static inline float reduce16(__m256 lo, __m256 hi) {   // _mm512_reduce_add_ps's order
    const __m256 t3 = _mm256_add_ps(hi, lo);
    const __m128 t6 = _mm_add_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
    const __m128 t8 = _mm_add_ps(t6, _mm_shuffle_ps(t6, t6, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(t8) + _mm_cvtss_f32(_mm_shuffle_ps(t8, t8, _MM_SHUFFLE(1, 1, 1, 1)));
}
// y[r * ldy + row] = sum_k W[row][k] x[r][k] for rows [0, N) of one matrix (codes q, fp16 d ds), x [M][K] bf16
template <int MF, int M, int R>
static inline void mat(const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
    const int KB = K / 256;
    const __m256i hm = _mm256_set1_epi32((int)0xffff0000);
    for (int row = 0; row < N; row += R) {
        __m256 acc[R][M][2];   // lanes 0-7 / 8-15 of the AVX-512 accumulator
        for (int i = 0; i < R; i++)
            for (int r = 0; r < M; r++) acc[i][r][0] = acc[i][r][1] = _mm256_setzero_ps();
        for (int b = 0; b < KB; b++) {
            Blk<MF> B[R];
            for (int i = 0; i < R; i++)
                B[i].prep(q + ((size_t)(row + i) * KB + b) * Blk<MF>::CB, _cvtsh_ss(ds[(size_t)(row + i) * KB + b]));
            for (int k = 0; k < 8; k++) {
                __m256 we[R][2], wo[R][2];   // weights of the lane pairs: lanes 0-7 = weights 0-15, 8-15 = 16-31
                for (int i = 0; i < R; i++) {
                    __m256 w4[4];
                    uint32_t s;
                    float el, eh;
                    B[i].w(k, w4, s, el, eh);
                    for (int h = 0; h < 4; h++) {
                        w4[h] = _mm256_mul_ps(w4[h], _mm256_set1_ps(h < 2 ? el : eh));
                        w4[h] = bf16r(_mm256_xor_ps(w4[h], sgn8((s >> (8 * h)) & 255)));
                    }
                    deint(w4[0], w4[1], we[i][0], wo[i][0]);
                    deint(w4[2], w4[3], we[i][1], wo[i][1]);
                }
                const int ko = b * 256 + 32 * k;
                for (int r = 0; r < M; r++) {
                    const uint16_t* xp = x + (size_t)r * K + ko;
                    for (int h = 0; h < 2; h++) {
                        const __m256i xv = _mm256_loadu_si256((const __m256i*)(xp + 16 * h));   // dword j = x[2j] | x[2j+1] << 16
                        const __m256 xe = _mm256_castsi256_ps(_mm256_slli_epi32(xv, 16));
                        const __m256 xo = _mm256_castsi256_ps(_mm256_and_si256(xv, hm));
                        for (int i = 0; i < R; i++)
                            acc[i][r][h] = _mm256_fmadd_ps(we[i][h], xe, _mm256_fmadd_ps(wo[i][h], xo, acc[i][r][h]));
                    }
                }
            }
        }
        for (int i = 0; i < R; i++)
            for (int r = 0; r < M; r++) y[(size_t)r * ldy + row + i] = reduce16(acc[i][r][0], acc[i][r][1]);
    }
}
template <int MF, int M>
static inline void mat_m(const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
    if constexpr (M <= 2) mat<MF, M, 2>(q, ds, N, K, x, y, ldy);   // (16 ymm: fewer weight rows per pass than AVX-512)
    else mat<MF, M, 1>(q, ds, N, K, x, y, ldy);
}
template <int MF>
static inline void mat_any(int m, const uint8_t* q, const uint16_t* ds, int N, int K, const uint16_t* x, float* y, int ldy) {
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
static inline void mat_run(int mf, int m, const uint8_t* q, const void* s, int N, int K, const uint16_t* x, float* y, int ldy) {
    const uint16_t* ds = (const uint16_t*)s;
    if (mf == XF_IQ2XXS) mat_any<XF_IQ2XXS>(m, q, ds, N, K, x, y, ldy);
    else if (mf == XF_IQ3XXS) mat_any<XF_IQ3XXS>(m, q, ds, N, K, x, y, ldy);
    else if (mf == XF_IQ2S) mat_any<XF_IQ2S>(m, q, ds, N, K, x, y, ldy);
    else mat_any<XF_IQ3S>(m, q, ds, N, K, x, y, ldy);
}
// GPU-equal helpers (as gguf_pfcpu.h)
static inline float e4m3_rne(float v) {
    const float a = fabsf(v);
    if (a == 0.f) return v;
    int ex;
    frexpf(a, &ex);
    const float qn = ldexpf(1.f, std::max(ex - 1, -6) - 3);
    return copysignf(nearbyintf(a / qn) * qn, v);
}
static inline float pow2ceil(float v) {
    int ex;
    const float mm = frexpf(v, &ex);
    if (mm == 0.5f) ex -= 1;
    return ldexpf(1.f, ex);
}
static inline uint16_t bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static inline void expert(XFmt f, const uint8_t* base, int m, const uint16_t* xg, const float* wt, float* h13, uint16_t* hq, float* Y) {
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

}  // namespace pfc2
}  // namespace dsv4
