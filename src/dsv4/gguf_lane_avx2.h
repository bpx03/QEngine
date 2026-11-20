#pragma once
// [dsv4-rel3-isa] AVX2 (x86-64-v3) build of gguf_lane.h: the same functions, signatures and per-output bits.
// The AVX-512 kernels keep 16 fp32 lanes per step; here the 16 lanes are two ymm halves (lanes 0-7, 8-15). Every lane
// gets the same integer dot (integer sums are exact in any order: u8 grid magnitudes <= 62, int8 activations, so
// vpmaddubsw pair sums stay below 2^15), the same float scale (same scalar expressions), the same fmadd per step and
// the reduction in _mm512_reduce_add_ps's order (GCC avx512fintrin.h: hi8 + lo8, hi4 + lo4, [0]+[2] / [1]+[3], sum).
// Activations stay in plain weight order: gg_perm_act copies (the plane kernels below read the plain order and
// build the plane kernels' lane sums from it), so the callers (dsv4_cpu_isa.cpp) are unchanged.
// Lane map of the AVX-512 kernels (per 256-weight block, dword lane k):
//   gathers (gg_row_dot), step m of 64 weights: lane k = weights 64 m + 4 k .. + 3 (one fmadd per m)
//   p3 / p3s (IQ3_XXS / IQ3_S planes): lane k = weights 16 k .. 16 k + 15 (one fmadd per block)
//   p2 (IQ2_XXS planes, two blocks per step): lane k < 8 = sub-block k of the first block (32 weights), k >= 8 = the
//      second block's sub-block k - 8 (one fmadd per block pair)
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gguf_iq.h"

namespace dsv4 {

namespace gl {
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

// _mm512_reduce_add_ps of the 16 lanes (lo = lanes 0-7, hi = lanes 8-15), same order
static inline float reduce16(__m256 lo, __m256 hi) {
    const __m256 t3 = _mm256_add_ps(hi, lo);
    const __m128 t6 = _mm_add_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
    const __m128 t8 = _mm_add_ps(t6, _mm_shuffle_ps(t6, t6, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(t8) + _mm_cvtss_f32(_mm_shuffle_ps(t8, t8, _MM_SHUFFLE(1, 1, 1, 1)));
}
// u8 x s8, 4 products per int32 lane (vpdpbusd's grouping without the accumulator)
static inline __m256i dot4(__m256i g, __m256i a) { return _mm256_madd_epi16(_mm256_maddubs_epi16(g, a), _mm256_set1_epi16(1)); }
// 32 sign bits (bit i = weight i) -> 0xFF bytes
static inline __m256i smask32(uint32_t s) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)s),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}
// lanes with a sign bit negated, saturating (as _mm512_mask_subs_epi8: a -128 stays in range, review L10)
// [dsv4-lane-avx2tune] (a ^ m) -s m: m = 0 -> a; m = -1 -> ~a + 1 saturating = -a, and -128 -> 127 (= subs(0, a))
// precondition (review rel5 L2): every byte of m is 0x00 or 0xFF (smask32 / kmask4); a sign-bit-only mask is wrong here
static inline __m256i sneg(__m256i a, __m256i m) { return _mm256_subs_epi8(_mm256_xor_si256(a, m), m); }
// [dsv4-lane-avx2tune] gg_ksign(i) (7 sign bits + parity) as 8 bytes of 0x00 / 0xFF, for the IQ2_XXS / IQ3_XXS masks
namespace {   // review rel5 L3: no ODR clash with another unit's names
struct KsMask {   // constexpr: no start-up code in this (AVX2) object
    uint64_t v[128];
};
static constexpr KsMask ksmask_make() {
    KsMask t{};
    for (int i = 0; i < 128; i++) {
        int pc = 0;
        for (int b = 0; b < 7; b++) pc += i >> b & 1;
        const unsigned s = (unsigned)i | (unsigned)(pc & 1) << 7;   // gg_ksign
        uint64_t x = 0;
        for (int b = 0; b < 8; b++)
            if (s >> b & 1) x |= 0xFFull << (8 * b);
        t.v[i] = x;
    }
    return t;
}
alignas(64) static constexpr KsMask g_ksmask = ksmask_make();
}  // namespace
static inline __m256i kmask4(uint32_t aux) {   // the masks of gg_ksign (gguf_iq.h) of the 7-bit fields at bits 0 / 7 / 14 / 21
    return _mm256_setr_epi64x((long long)g_ksmask.v[aux & 127], (long long)g_ksmask.v[(aux >> 7) & 127],
                              (long long)g_ksmask.v[(aux >> 14) & 127], (long long)g_ksmask.v[(aux >> 21) & 127]);
}
// per lane: the sum of the 8 dwords of v[j] -> lane j (8 vectors -> 8 lanes)
static inline __m256i hsum8x8(const __m256i* v) {
    const __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));   // [v0 v1 v2 v3 | v0 v1 v2 v3] halves
    const __m256i b = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31));
}
// per lane: lane k = the sum of dwords 4k .. 4k + 3 of v[0..3] (32 dwords -> 8 lanes)
static inline __m256i hsum4x4(const __m256i* v) {
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));   // L0 L2 L4 L6 | L1 L3 L5 L7
    return _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
static inline uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
// IQ2_XXS: the aux dword (signs, scale) of each of the 8 sub-blocks
static inline __m256i aux8(const uint8_t* c) {
    return _mm256_setr_epi32((int)ld32(c + 4), (int)ld32(c + 12), (int)ld32(c + 20), (int)ld32(c + 28), (int)ld32(c + 36),
                             (int)ld32(c + 44), (int)ld32(c + 52), (int)ld32(c + 60));
}

// one super-block (codes c, engine or GGUF layout: same code bytes) -> the 8 sub-blocks' grid magnitudes g[k] (32 u8
// in weight order) and sign masks m[k]
template <int MF>
static inline void blk_decode(const uint8_t* c, __m256i* g, __m256i* m) {
    if constexpr (MF == XF_IQ2XXS) {
        for (int k = 0; k < 8; k++) {
            const uint8_t* q = c + 8 * k;
            g[k] = _mm256_setr_epi64x((long long)gg_iq2xxs_grid[q[0]], (long long)gg_iq2xxs_grid[q[1]],
                                      (long long)gg_iq2xxs_grid[q[2]], (long long)gg_iq2xxs_grid[q[3]]);
            m[k] = kmask4(ld32(q + 4));
        }
    } else if constexpr (MF == XF_IQ3XXS) {
        for (int k = 0; k < 8; k++) {
            const uint8_t* q = c + 8 * k;   // (scalar loads, no vpgatherdd: slow on Intel with the GDS fix; review rel3 L5)
            g[k] = _mm256_setr_epi32((int)gg_iq3xxs_grid[q[0]], (int)gg_iq3xxs_grid[q[1]], (int)gg_iq3xxs_grid[q[2]], (int)gg_iq3xxs_grid[q[3]],
                                     (int)gg_iq3xxs_grid[q[4]], (int)gg_iq3xxs_grid[q[5]], (int)gg_iq3xxs_grid[q[6]], (int)gg_iq3xxs_grid[q[7]]);
            m[k] = kmask4(ld32(c + 64 + 4 * k));
        }
    } else if constexpr (MF == XF_IQ2S) {
        for (int k = 0; k < 8; k++) {
            const uint8_t* q = c + 4 * k;
            const uint32_t h = c[64 + k];
            g[k] = _mm256_setr_epi64x((long long)gg_iq2s_grid[q[0] | ((h & 3) << 8)], (long long)gg_iq2s_grid[q[1] | (((h >> 2) & 3) << 8)],
                                      (long long)gg_iq2s_grid[q[2] | (((h >> 4) & 3) << 8)], (long long)gg_iq2s_grid[q[3] | (((h >> 6) & 3) << 8)]);
            m[k] = smask32(ld32(c + 32 + 4 * k));
        }
    } else {   // XF_IQ3S: 9-bit indices qs[i] | ((qh[k] >> i) & 1) << 8
        for (int k = 0; k < 8; k++) {
            const uint8_t* q = c + 8 * k;
            const uint32_t h = c[64 + k];
            auto G = [&](int i) { return (int)gg_iq3s_grid[q[i] | (((h >> i) & 1) << 8)]; };
            g[k] = _mm256_setr_epi32(G(0), G(1), G(2), G(3), G(4), G(5), G(6), G(7));
            m[k] = smask32(ld32(c + 72 + 4 * k));
        }
    }
}
// dv[k] = the 8 dword dots (4 weights each) of sub-block k with activations a (256, plain order)
static inline void blk_dots(const __m256i* g, const __m256i* m, const int8_t* a, __m256i* dv) {
    for (int k = 0; k < 8; k++) dv[k] = dot4(g[k], sneg(_mm256_loadu_si256((const __m256i*)(a + 32 * k)), m[k]));
}
}  // namespace gl

// ---------------------------------------------------------------- activation permutation (plane order)

constexpr int GG_PERM_MAXK = 8192;   // review L9: the plane path is taken only up to this K (callers' buffers)
static inline int gg_planes_on() {
    static const int on = getenv("DSV4_GG_GATHER") ? 0 : 1;   // 1 = planes where available (R16 default)
    return on;
}
static inline int gg_plane_kind(int mf, int K) {
    if (!gg_planes_on() || K > GG_PERM_MAXK || K % 256) return 0;
    if (mf == XF_IQ2XXS) return (K / 256) % 2 == 0 ? 2 : 0;
    if (mf == XF_IQ3XXS || mf == XF_IQ3S) return 3;
    return 0;
}
// AVX2 build: the plane kernels read the plain order, so the "plane-order copy" is a copy
static inline void gg_perm_act(int kind, const int8_t* a, int K, int8_t* ap) {
    (void)kind;
    memcpy(ap, a, (size_t)K);
}

// ---------------------------------------------------------------- gather kernels (one row)

template <int MF, bool NAT = false>
static inline float gg_row_dot(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* a, const float* ascl) {
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    constexpr int CS = NAT ? CB + 2 : CB, DS = NAT ? CS / 2 : 1;
    for (int b = 0; b < nb; b++, c += CS, a += 256) {
        const float d = gl::h2f(ds[b * DS]);
        const float da0 = d * ascl[2 * b], da1 = d * ascl[2 * b + 1];
        __m256i g[8], m[8], dv[8];
        gl::blk_decode<MF>(c, g, m);
        gl::blk_dots(g, m, a, dv);
        for (int s = 0; s < 4; s++) {   // AVX-512 step m = s: sub-blocks 2 s (lanes 0-7) and 2 s + 1 (lanes 8-15)
            __m256 f0, f1;
            if constexpr (MF == XF_IQ2XXS) {
                const float fa = (float)(2 * (int)(gl::ld32(c + 16 * s + 4) >> 28) + 1) * 0.125f;
                const float fb = (float)(2 * (int)(gl::ld32(c + 16 * s + 12) >> 28) + 1) * 0.125f;
                const float da = s < 2 ? da0 : da1;
                f0 = _mm256_set1_ps(fa * da);
                f1 = _mm256_set1_ps(fb * da);
            } else if constexpr (MF == XF_IQ3XXS) {
                const float da = (s < 2 ? da0 : da1) * 0.25f;
                f0 = _mm256_set1_ps((float)(2 * (int)(gl::ld32(c + 64 + 8 * s) >> 28) + 1) * da);
                f1 = _mm256_set1_ps((float)(2 * (int)(gl::ld32(c + 68 + 8 * s) >> 28) + 1) * da);
            } else if constexpr (MF == XF_IQ2S) {
                const float da = (s < 2 ? da0 : da1) * 0.125f;
                const uint8_t s0 = c[72 + 2 * s], s1 = c[72 + 2 * s + 1];
                const float x0 = (float)(2 * (s0 & 15) + 1) * da, x1 = (float)(2 * (s0 >> 4) + 1) * da;
                const float x2 = (float)(2 * (s1 & 15) + 1) * da, x3 = (float)(2 * (s1 >> 4) + 1) * da;
                f0 = _mm256_setr_ps(x0, x0, x0, x0, x1, x1, x1, x1);
                f1 = _mm256_setr_ps(x2, x2, x2, x2, x3, x3, x3, x3);
            } else {
                const float da = s < 2 ? da0 : da1;
                const uint8_t sc = c[104 + s];
                f0 = _mm256_set1_ps((float)(1 + 2 * (sc & 15)) * da);
                f1 = _mm256_set1_ps((float)(1 + 2 * (sc >> 4)) * da);
            }
            lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dv[2 * s]), f0, lo);
            hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dv[2 * s + 1]), f1, hi);
        }
    }
    return gl::reduce16(lo, hi);
}

// ---------------------------------------------------------------- plane kernels (one row; ap = plain-order activations)

// IQ2_XXS, two blocks per step (nb even)
template <bool NAT = false>
static inline float gg_row_dot_p2(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                   const uint8_t* pf = nullptr) {
    constexpr int CS = NAT ? 66 : 64, DS = NAT ? 33 : 1;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int p = 0; p < nb / 2; p++, c += 2 * CS, ap += 512) {
        if (pf) {
            _mm_prefetch((const char*)(pf + 2 * CS * p), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + 2 * CS * p + 64), _MM_HINT_T0);
            if (NAT) _mm_prefetch((const char*)(pf + 2 * CS * p + 128), _MM_HINT_T0);
        }
        __m256i g[8], m[8], dv[8];
        gl::blk_decode<XF_IQ2XXS>(c, g, m);
        gl::blk_dots(g, m, ap, dv);
        const __m256i dot0 = gl::hsum8x8(dv);
        gl::blk_decode<XF_IQ2XXS>(c + CS, g, m);
        gl::blk_dots(g, m, ap + 256, dv);
        const __m256i dot1 = gl::hsum8x8(dv);
        // scale bits 28..31 of each sub-block's aux dword (lane k)
        const __m256i aux0 = gl::aux8(c);
        const __m256i aux1 = gl::aux8(c + CS);
        const __m256 sc0 = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux0, 28), 1), _mm256_set1_epi32(1)));
        const __m256 sc1 = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux1, 28), 1), _mm256_set1_epi32(1)));
        const float d0 = gl::h2f(ds[2 * p * DS]) * 0.125f, d1 = gl::h2f(ds[(2 * p + 1) * DS]) * 0.125f;
        const float f0 = d0 * ascl[4 * p], f1 = d0 * ascl[4 * p + 1], f2 = d1 * ascl[4 * p + 2], f3 = d1 * ascl[4 * p + 3];
        const __m256 fl = _mm256_mul_ps(sc0, _mm256_setr_ps(f0, f0, f0, f0, f1, f1, f1, f1));
        const __m256 fh = _mm256_mul_ps(sc1, _mm256_setr_ps(f2, f2, f2, f2, f3, f3, f3, f3));
        lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), fl, lo);
        hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1), fh, hi);
    }
    return gl::reduce16(lo, hi);
}
// IQ3_XXS / IQ3_S plane lanes: lane k = 16 weights; scale (2 s + 1) of sub-block k / 2
template <int MF>
static inline void p3_lanes(const __m256i* dv, __m256i& dl, __m256i& dh) {
    dl = gl::hsum4x4(dv);
    dh = gl::hsum4x4(dv + 4);
}
// IQ3_XXS, one block per step
template <bool NAT = false>
static inline float gg_row_dot_p3(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                   const uint8_t* pf = nullptr) {
    constexpr int CS = NAT ? 98 : 96, DS = NAT ? 49 : 1;
    const __m256i dsel = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS, ap += 256) {
        if (pf) {
            _mm_prefetch((const char*)(pf + CS * b), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + CS * b + 64), _MM_HINT_T0);
        }
        __m256i g[8], m[8], dv[8], dl, dh;
        gl::blk_decode<XF_IQ3XXS>(c, g, m);
        gl::blk_dots(g, m, ap, dv);
        p3_lanes<XF_IQ3XXS>(dv, dl, dh);
        const __m256i sw = _mm256_loadu_si256((const __m256i*)(c + 64));   // aux dword per sub-block
        const __m256i sl = _mm256_srli_epi32(_mm256_permutevar8x32_epi32(sw, dsel), 28);
        const __m256i sh = _mm256_srli_epi32(_mm256_permutevar8x32_epi32(sw, _mm256_add_epi32(dsel, _mm256_set1_epi32(4))), 28);
        const float d = gl::h2f(ds[b * DS]) * 0.25f, da0 = d * ascl[2 * b], da1 = d * ascl[2 * b + 1];
        const __m256 fl = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(sl, 1), _mm256_set1_epi32(1))), _mm256_set1_ps(da0));
        const __m256 fh = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(sh, 1), _mm256_set1_epi32(1))), _mm256_set1_ps(da1));
        lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), fl, lo);
        hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), fh, hi);
    }
    return gl::reduce16(lo, hi);
}
// IQ3_S, one block per step
template <bool NAT = false>
static inline float gg_row_dot_p3s(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                    const uint8_t* pf = nullptr) {
    constexpr int CS = NAT ? 110 : 108, DS = NAT ? 55 : 1;
    const __m256i nsl = _mm256_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12), nsh = _mm256_setr_epi32(16, 16, 20, 20, 24, 24, 28, 28);
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS, ap += 256) {
        if (pf) {
            _mm_prefetch((const char*)(pf + CS * b), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + CS * b + 64), _MM_HINT_T0);
        }
        __m256i g[8], m[8], dv[8], dl, dh;
        gl::blk_decode<XF_IQ3S>(c, g, m);
        gl::blk_dots(g, m, ap, dv);
        p3_lanes<XF_IQ3S>(dv, dl, dh);
        const __m256i s4 = _mm256_set1_epi32((int)gl::ld32(c + 104));
        const __m256i nl = _mm256_and_si256(_mm256_srlv_epi32(s4, nsl), _mm256_set1_epi32(15));
        const __m256i nh = _mm256_and_si256(_mm256_srlv_epi32(s4, nsh), _mm256_set1_epi32(15));
        const float d = gl::h2f(ds[b * DS]);
        const __m256 fl = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(nl, 1), _mm256_set1_epi32(1))), _mm256_set1_ps(d * ascl[2 * b]));
        const __m256 fh = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(nh, 1), _mm256_set1_epi32(1))), _mm256_set1_ps(d * ascl[2 * b + 1]));
        lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), fl, lo);
        hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), fh, hi);
    }
    return gl::reduce16(lo, hi);
}

// [dsv4-dec1-lmr] NR activation rows against one weight row: the weight decode once, the signed dot per row; each
// row's float sequence is the one-row kernel's (bit-identical per row)
template <int NR, bool NAT = false>
static inline void gg_rows_dot_p2(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* const* ap,
                                  const float* const* ascl, float* out) {
    constexpr int CS = NAT ? 66 : 64, DS = NAT ? 33 : 1;
    __m256 lo[NR], hi[NR];
    for (int i = 0; i < NR; i++) lo[i] = hi[i] = _mm256_setzero_ps();
    for (int p = 0; p < nb / 2; p++, c += 2 * CS) {
        __m256i g0[8], m0[8], g1[8], m1[8], dv[8];
        gl::blk_decode<XF_IQ2XXS>(c, g0, m0);
        gl::blk_decode<XF_IQ2XXS>(c + CS, g1, m1);
        const __m256i aux0 = gl::aux8(c);
        const __m256i aux1 = gl::aux8(c + CS);
        const __m256 sc0 = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux0, 28), 1), _mm256_set1_epi32(1)));
        const __m256 sc1 = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux1, 28), 1), _mm256_set1_epi32(1)));
        const float d0 = gl::h2f(ds[2 * p * DS]) * 0.125f, d1 = gl::h2f(ds[(2 * p + 1) * DS]) * 0.125f;
        for (int i = 0; i < NR; i++) {
            const float* as = ascl[i];
            const float f0 = d0 * as[4 * p], f1 = d0 * as[4 * p + 1], f2 = d1 * as[4 * p + 2], f3 = d1 * as[4 * p + 3];
            const __m256 fl = _mm256_mul_ps(sc0, _mm256_setr_ps(f0, f0, f0, f0, f1, f1, f1, f1));
            const __m256 fh = _mm256_mul_ps(sc1, _mm256_setr_ps(f2, f2, f2, f2, f3, f3, f3, f3));
            const int8_t* a = ap[i] + 512 * p;
            gl::blk_dots(g0, m0, a, dv);
            const __m256i dot0 = gl::hsum8x8(dv);
            gl::blk_dots(g1, m1, a + 256, dv);
            const __m256i dot1 = gl::hsum8x8(dv);
            lo[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), fl, lo[i]);
            hi[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1), fh, hi[i]);
        }
    }
    for (int i = 0; i < NR; i++) out[i] = gl::reduce16(lo[i], hi[i]);
}
template <int NR, bool NAT = false>
static inline void gg_rows_dot_p3(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* const* ap,
                                  const float* const* ascl, float* out) {
    constexpr int CS = NAT ? 98 : 96, DS = NAT ? 49 : 1;
    const __m256i dsel = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    __m256 lo[NR], hi[NR];
    for (int i = 0; i < NR; i++) lo[i] = hi[i] = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS) {
        __m256i g[8], m[8], dv[8], dl, dh;
        gl::blk_decode<XF_IQ3XXS>(c, g, m);
        const __m256i sw = _mm256_loadu_si256((const __m256i*)(c + 64));
        const __m256i sl = _mm256_srli_epi32(_mm256_permutevar8x32_epi32(sw, dsel), 28);
        const __m256i sh = _mm256_srli_epi32(_mm256_permutevar8x32_epi32(sw, _mm256_add_epi32(dsel, _mm256_set1_epi32(4))), 28);
        const __m256 scl = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(sl, 1), _mm256_set1_epi32(1)));
        const __m256 sch = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(sh, 1), _mm256_set1_epi32(1)));
        const float d = gl::h2f(ds[b * DS]) * 0.25f;
        for (int i = 0; i < NR; i++) {
            const float da0 = d * ascl[i][2 * b], da1 = d * ascl[i][2 * b + 1];
            const __m256 fl = _mm256_mul_ps(scl, _mm256_set1_ps(da0)), fh = _mm256_mul_ps(sch, _mm256_set1_ps(da1));
            gl::blk_dots(g, m, ap[i] + 256 * b, dv);
            p3_lanes<XF_IQ3XXS>(dv, dl, dh);
            lo[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), fl, lo[i]);
            hi[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), fh, hi[i]);
        }
    }
    for (int i = 0; i < NR; i++) out[i] = gl::reduce16(lo[i], hi[i]);
}

// ---------------------------------------------------------------- matrix rows (as gguf_lane.h)

static inline int gg_pfd() {
    static const int v = getenv("DSV4_GGPFD") ? atoi(getenv("DSV4_GGPFD")) : 0;
    return v;
}
static inline void k_ggv_any(int mf, const uint8_t* q, const uint16_t* s, int K, int r0, int r1, const int8_t* a,
                             const float* ascl, float* y, const int8_t* ap = nullptr) {
    const int nb = K / 256, cb = gg_cbytes(mf);
    const int kind = gg_plane_kind(mf, K);
    if (kind) {
        if (!ap) ap = a;   // (plain order here)
        const int pfd = gg_pfd();
        for (int r = r0; r < r1; r++) {
            const uint8_t* c = q + (size_t)r * nb * cb;
            const uint16_t* d = s + (size_t)r * nb;
            const uint8_t* pf = pfd && r + pfd < r1 ? c + (size_t)pfd * nb * cb : nullptr;
            if (pf) _mm_prefetch((const char*)(d + (size_t)pfd * nb), _MM_HINT_T0);
            y[r] = mf == XF_IQ2XXS ? gg_row_dot_p2(c, d, nb, ap, ascl, pf)
                 : mf == XF_IQ3XXS ? gg_row_dot_p3(c, d, nb, ap, ascl, pf)
                                   : gg_row_dot_p3s(c, d, nb, ap, ascl, pf);
        }
        return;
    }
    for (int r = r0; r < r1; r++) {
        const uint8_t* c = q + (size_t)r * nb * cb;
        const uint16_t* d = s + (size_t)r * nb;
        switch (mf) {
            case XF_IQ2XXS: y[r] = gg_row_dot<XF_IQ2XXS>(c, d, nb, a, ascl); break;
            case XF_IQ3XXS: y[r] = gg_row_dot<XF_IQ3XXS>(c, d, nb, a, ascl); break;
            case XF_IQ2S: y[r] = gg_row_dot<XF_IQ2S>(c, d, nb, a, ascl); break;
            default: y[r] = gg_row_dot<XF_IQ3S>(c, d, nb, a, ascl); break;
        }
    }
}
static inline bool k_ggv_rows(int mf, bool nat, const uint8_t* q, const uint16_t* s, int K, int r0, int r1, int n,
                              const int8_t* const* ap, const float* const* ascl, float* const* y) {
    const int kind = gg_plane_kind(mf, K);
    if (n < 2 || n > 6 || !((mf == XF_IQ2XXS && kind == 2) || (mf == XF_IQ3XXS && kind == 3))) return false;
    for (int i = 0; i < n; i++)
        if (!ap[i]) return false;
    const int nb = K / 256, cb = gg_cbytes(mf);
    float o[6];
    for (int r = r0; r < r1; r++) {
        const uint8_t* c;
        const uint16_t* d;
        if (nat) {
            const uint8_t* row = q + (size_t)r * nb * (cb + 2);
            d = (const uint16_t*)row;
            c = row + 2;
        } else {
            c = q + (size_t)r * nb * cb;
            d = s + (size_t)r * nb;
        }
#define LMR_CASE(NN)                                                                                    \
    case NN:                                                                                            \
        if (mf == XF_IQ2XXS) nat ? gg_rows_dot_p2<NN, true>(c, d, nb, ap, ascl, o) : gg_rows_dot_p2<NN>(c, d, nb, ap, ascl, o); \
        else nat ? gg_rows_dot_p3<NN, true>(c, d, nb, ap, ascl, o) : gg_rows_dot_p3<NN>(c, d, nb, ap, ascl, o);            \
        break;
        switch (n) { LMR_CASE(2) LMR_CASE(3) LMR_CASE(4) LMR_CASE(5) LMR_CASE(6) }
#undef LMR_CASE
        for (int i = 0; i < n; i++) y[i][r] = o[i];
    }
    return true;
}
static inline void k_ggv_nat(int mf, const uint8_t* raw, int K, int r0, int r1, const int8_t* a, const float* ascl,
                             float* y, const int8_t* ap = nullptr) {
    const int nb = K / 256, bb = gg_cbytes(mf) + 2;
    const int kind = gg_plane_kind(mf, K);
    if (kind) {
        if (!ap) ap = a;
        const int pfd = gg_pfd();
        for (int r = r0; r < r1; r++) {
            const uint8_t* row = raw + (size_t)r * nb * bb;
            const uint16_t* d = (const uint16_t*)row;
            const uint8_t* pf = pfd && r + pfd < r1 ? row + 2 + (size_t)pfd * nb * bb : nullptr;
            y[r] = mf == XF_IQ2XXS ? gg_row_dot_p2<true>(row + 2, d, nb, ap, ascl, pf)
                 : mf == XF_IQ3XXS ? gg_row_dot_p3<true>(row + 2, d, nb, ap, ascl, pf)
                                   : gg_row_dot_p3s<true>(row + 2, d, nb, ap, ascl, pf);
        }
        return;
    }
    for (int r = r0; r < r1; r++) {
        const uint8_t* row = raw + (size_t)r * nb * bb;
        const uint16_t* d = (const uint16_t*)row;
        switch (mf) {
            case XF_IQ2XXS: y[r] = gg_row_dot<XF_IQ2XXS, true>(row + 2, d, nb, a, ascl); break;
            case XF_IQ3XXS: y[r] = gg_row_dot<XF_IQ3XXS, true>(row + 2, d, nb, a, ascl); break;
            case XF_IQ2S: y[r] = gg_row_dot<XF_IQ2S, true>(row + 2, d, nb, a, ascl); break;
            default: y[r] = gg_row_dot<XF_IQ3S, true>(row + 2, d, nb, a, ascl); break;
        }
    }
}

}  // namespace dsv4
