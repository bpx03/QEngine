#pragma once
// R16 [dsv4-gguf-iq]: CPU lane GEMV for the GGUF i-quant matrix formats (gguf_iq.h layout), AVX-512 VNNI + VBMI +
// BITALG (Zen 4). y[r] = row r . activation, rows [r0, r1), one activation row: int8 with a scale per 128 (the
// lane's plain layout, dsv4_cpu.cpp quant_row). Grid magnitudes are unsigned bytes (u8 operand of vpdpbusd), the
// signs go onto the int8 activations (masked saturating negate), integer dot per 4 weights, scale (super-block d x
// sub-block scale x activation scale) per 32 / 16 weights in fp32.
// Two lookup schemes:
//   planes  (IQ2_XXS with an even number of blocks per row, IQ3_XXS, IQ3_S): the grid split into byte planes (plane j
//           = byte j of every entry, 256 B = 4 zmm; IQ3_S's 512 entries = two halves + a blend on the 9th index bit),
//           64 entries per vpermi2b pair + top-bit blend. The activations are permuted once per (row, matrix kind)
//           into plane order (plane j, lane i = weight W i + j of the entry in lane i): gg_perm_act, cached by the
//           caller (dsv4_cpu.cpp: once per row and run, review M6).
//   gathers (IQ2_S, and the rest): 8 entries of 8 B (IQ2) or 16 of 4 B (IQ3) per gather = 64 weights; the sign bits
//           are already in weight order, so the activations need no permutation.
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>

#include "gguf_iq.h"

namespace dsv4 {

namespace gl {
namespace {   // [dsv4-rel3-isa] internal linkage (Planes' constructor too)
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

// bytes 8k + l (k 0..7, l 0..3) of a 64-byte vector -> bytes 0..31
static inline __m512i perm_q4() {
    alignas(64) static const uint8_t p[64] = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
                                              32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59};
    return _mm512_load_si512(p);
}
static inline __m512i parity7(__m512i s) {   // s & 0x7f with bit 7 = parity of bits 0..6
    s = _mm512_and_si512(s, _mm512_set1_epi8(0x7f));
    const __m512i par = _mm512_and_si512(_mm512_popcnt_epi8(s), _mm512_set1_epi8(1));
    return _mm512_or_si512(s, _mm512_slli_epi16(par, 7));
}
// four 7-bit sign fields at bit 0 / 7 / 14 / 21 of the upper (IQ2_XXS: off 32) or lower (off 0) dword of every
// qword -> 8-bit signs with parity, 32 bytes in weight order (field (k, l) -> byte 4k + l)
static inline void signs32(__m512i v, int off, uint64_t sm[4]) {
    const __m512i ctrl = _mm512_set1_epi64((long long)((uint64_t)(off + 0) | ((uint64_t)(off + 7) << 8) |
                                                       ((uint64_t)(off + 14) << 16) | ((uint64_t)(off + 21) << 24)));
    const __m512i s = _mm512_permutexvar_epi8(perm_q4(), parity7(_mm512_multishift_epi64_epi8(ctrl, v)));
    _mm256_storeu_si256((__m256i*)sm, _mm512_castsi512_si256(s));
}
static inline __m512 two(float a, float b) {   // lanes 0-7 = a, 8-15 = b
    return _mm512_insertf32x8(_mm512_set1_ps(a), _mm256_set1_ps(b), 1);
}
// signed activations: lanes with a sign bit negated (saturating: a -128 from a degenerate group stays in range,
// review L10), then u8 grid x s8 activation
static inline __m512i sdot(__m512i acc, __m512i g, __m512i av, __mmask64 m) {
    return _mm512_dpbusd_epi32(acc, g, _mm512_mask_subs_epi8(av, m, _mm512_setzero_si512(), av));
}

struct Planes {
    alignas(64) uint8_t p2[8][256];    // IQ2_XXS: 8 planes
    alignas(64) uint8_t p3[4][256];    // IQ3_XXS: 4 planes
    alignas(64) uint8_t p3s[4][512];   // IQ3_S: 4 planes of 512
    Planes() {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 8; j++) p2[j][i] = (uint8_t)(gg_iq2xxs_grid[i] >> (8 * j));
            for (int j = 0; j < 4; j++) p3[j][i] = (uint8_t)(gg_iq3xxs_grid[i] >> (8 * j));
        }
        for (int i = 0; i < 512; i++)
            for (int j = 0; j < 4; j++) p3s[j][i] = (uint8_t)(gg_iq3s_grid[i] >> (8 * j));
    }
};
static const Planes& planes() {
    static const Planes P;
    return P;
}
static inline __m512i plane_lookup(const uint8_t* T, __m512i idx, __mmask64 top) {   // 256-entry plane
    const __m512i lo = _mm512_permutex2var_epi8(_mm512_load_si512(T), idx, _mm512_load_si512(T + 64));
    const __m512i hi = _mm512_permutex2var_epi8(_mm512_load_si512(T + 128), idx, _mm512_load_si512(T + 192));
    return _mm512_mask_blend_epi8(top, lo, hi);
}
}  // namespace
}  // namespace gl

// ---------------------------------------------------------------- activation permutation (plane order)

constexpr int GG_PERM_MAXK = 8192;   // review L9: the plane path is taken only up to this K (callers' buffers)
// 0: no plane path; 2: IQ2_XXS order (two blocks per step, 8 planes); 3: IQ3_XXS / IQ3_S order (4 planes)
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
// kind 2: per pair of blocks (512 B = 64 entries x 8 B): plane j lane i = byte j of entry i; kind 3: per block
// (256 B = 64 entries x 4 B): plane j lane i = byte j of entry i
static inline void gg_perm_act(int kind, const int8_t* a, int K, int8_t* ap) {
    if (kind == 2) {
        alignas(64) static const uint8_t pq[64] = {   // within a zmm of 8 entries: byte (8 e + j) -> (8 j + e)
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
            4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};
        const __m512i P = _mm512_load_si512(pq);
        const __m512i e0 = _mm512_setr_epi64(0, 8, 2, 10, 4, 12, 6, 14), o0 = _mm512_setr_epi64(1, 9, 3, 11, 5, 13, 7, 15);
        const __m512i e1 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13), o1 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
        const __m512i e2 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11), o2 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        for (int p = 0; p < K / 512; p++) {
            __m512i r[8];
            for (int z = 0; z < 8; z++) r[z] = _mm512_permutexvar_epi8(P, _mm512_loadu_si512(a + p * 512 + 64 * z));
            // 8 x 8 qword transpose: out[j] = qword j of r[0..7]
            __m512i a1[8], a2[8], a3[8];
            for (int k = 0; k < 4; k++) {
                a1[2 * k] = _mm512_permutex2var_epi64(r[2 * k], e0, r[2 * k + 1]);       // q0 q2 q4 q6 of the pair
                a1[2 * k + 1] = _mm512_permutex2var_epi64(r[2 * k], o0, r[2 * k + 1]);   // q1 q3 q5 q7
            }
            for (int k = 0; k < 2; k++)
                for (int h = 0; h < 2; h++) {
                    a2[4 * k + 2 * h] = _mm512_permutex2var_epi64(a1[4 * k + h], e1, a1[4 * k + 2 + h]);       // q0 / q1 x4, q4 / q5 x4
                    a2[4 * k + 2 * h + 1] = _mm512_permutex2var_epi64(a1[4 * k + h], o1, a1[4 * k + 2 + h]);   // q2 / q3 x4, q6 / q7 x4
                }
            for (int t = 0; t < 4; t++) {
                a3[2 * t] = _mm512_permutex2var_epi64(a2[t], e2, a2[4 + t]);
                a3[2 * t + 1] = _mm512_permutex2var_epi64(a2[t], o2, a2[4 + t]);
            }
            // a2[0] = q0 x4 | q4 x4 (k 0: entries rows 0-3), a2[1] = q2 | q6, a2[2] = q1 | q5, a2[3] = q3 | q7
            // a3[2t] = low halves of a2[t] (k 0) + of a2[4+t] (k 1) = one plane, a3[2t+1] = the other
            static const int plane_of[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            for (int i = 0; i < 8; i++) _mm512_storeu_si512(ap + p * 512 + 64 * plane_of[i], a3[i]);
        }
    } else {
        alignas(64) static const uint8_t pd[64] = {   // within a zmm of 16 entries: byte (4 e + j) -> (16 j + e)
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
            2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
        const __m512i P = _mm512_load_si512(pd);
        for (int b = 0; b < K / 256; b++) {
            __m512i r[4];
            for (int z = 0; z < 4; z++) r[z] = _mm512_permutexvar_epi8(P, _mm512_loadu_si512(a + b * 256 + 64 * z));
            const __m512i t0 = _mm512_shuffle_i64x2(r[0], r[1], 0x44), t1 = _mm512_shuffle_i64x2(r[0], r[1], 0xEE);
            const __m512i t2 = _mm512_shuffle_i64x2(r[2], r[3], 0x44), t3 = _mm512_shuffle_i64x2(r[2], r[3], 0xEE);
            _mm512_storeu_si512(ap + b * 256 + 0, _mm512_shuffle_i64x2(t0, t2, 0x88));
            _mm512_storeu_si512(ap + b * 256 + 64, _mm512_shuffle_i64x2(t0, t2, 0xDD));
            _mm512_storeu_si512(ap + b * 256 + 128, _mm512_shuffle_i64x2(t1, t3, 0x88));
            _mm512_storeu_si512(ap + b * 256 + 192, _mm512_shuffle_i64x2(t1, t3, 0xDD));
        }
    }
}
// scalar reference of the same permutation (tests)
static inline void gg_perm_act_ref(int kind, const int8_t* a, int K, int8_t* ap) {
    if (kind == 2) {
        for (int p = 0; p < K / 512; p++)
            for (int j = 0; j < 8; j++)
                for (int i = 0; i < 64; i++) ap[p * 512 + 64 * j + i] = a[(2 * p + i / 32) * 256 + 8 * (i % 32) + j];
    } else {
        for (int b = 0; b < K / 256; b++)
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 64; i++) ap[b * 256 + 64 * j + i] = a[b * 256 + 4 * i + j];
    }
}

// ---------------------------------------------------------------- gather kernels (one row)

// NAT [dsv4-rel-ram]: the GGUF's own block layout (d, then the code bytes, per 256 weights: c = row + 2, ds = row,
// block stride CB + 2 bytes); else the engine's re-pack (codes, d separately). Same arithmetic either way.
template <int MF, bool NAT = false>
static inline float gg_row_dot(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* a, const float* ascl) {
    __m512 acc = _mm512_setzero_ps();
    constexpr int CB = MF == XF_IQ2XXS ? 64 : MF == XF_IQ3XXS ? 96 : MF == XF_IQ2S ? 80 : 108;
    constexpr int CS = NAT ? CB + 2 : CB, DS = NAT ? CS / 2 : 1;
    for (int b = 0; b < nb; b++, c += CS, a += 256) {
        const float d = gl::h2f(ds[b * DS]);
        const float da0 = d * ascl[2 * b], da1 = d * ascl[2 * b + 1];
        alignas(32) uint64_t sm[4];
        if constexpr (MF == XF_IQ2XXS) {
            const __m512i raw = _mm512_loadu_si512(c);
            gl::signs32(raw, 32, sm);
            alignas(64) uint8_t ix[64];
            _mm512_store_si512(ix, _mm512_permutexvar_epi8(gl::perm_q4(), raw));
            alignas(64) uint64_t q[8];
            _mm512_store_si512(q, raw);
            float f[8];
            for (int k = 0; k < 8; k++) f[k] = (float)(2 * (int)(q[k] >> 60) + 1) * 0.125f;
            for (int m = 0; m < 4; m++) {
                const __m256i idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ix + 8 * m)));
                const __m512i g = _mm512_i32gather_epi64(idx, (const long long*)gg_iq2xxs_grid, 8);
                const __m512i dt = gl::sdot(_mm512_setzero_si512(), g, _mm512_loadu_si512(a + 64 * m), sm[m]);
                const float da = m < 2 ? da0 : da1;
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dt), gl::two(f[2 * m] * da, f[2 * m + 1] * da), acc);
            }
        } else if constexpr (MF == XF_IQ3XXS) {
            const __m512i sw = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)(c + 64)));
            gl::signs32(sw, 0, sm);
            uint32_t w[8];
            memcpy(w, c + 64, 32);
            for (int m = 0; m < 4; m++) {
                const __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + 16 * m)));
                const __m512i g = _mm512_i32gather_epi32(idx, (const int*)gg_iq3xxs_grid, 4);
                const __m512i dt = gl::sdot(_mm512_setzero_si512(), g, _mm512_loadu_si512(a + 64 * m), sm[m]);
                const float da = (m < 2 ? da0 : da1) * 0.25f;
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dt),
                                      gl::two((float)(2 * (int)(w[2 * m] >> 28) + 1) * da, (float)(2 * (int)(w[2 * m + 1] >> 28) + 1) * da), acc);
            }
        } else if constexpr (MF == XF_IQ2S) {
            // 10-bit indices: qs[i] | ((qh[i / 4] >> 2 (i % 4)) & 3) << 8, i = 0..31
            const __m128i qh = _mm_loadl_epi64((const __m128i*)(c + 64));
            const __m512i sh = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
            alignas(64) uint32_t ix[32];
            for (int h = 0; h < 2; h++) {
                const __m128i hb = _mm_shuffle_epi8(qh, _mm_setr_epi8(4 * h, 4 * h, 4 * h, 4 * h, 4 * h + 1, 4 * h + 1, 4 * h + 1, 4 * h + 1,
                                                                      4 * h + 2, 4 * h + 2, 4 * h + 2, 4 * h + 2, 4 * h + 3, 4 * h + 3, 4 * h + 3, 4 * h + 3));
                __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(_mm512_cvtepu8_epi32(hb), sh), _mm512_set1_epi32(3));
                __m512i lo = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + 16 * h)));
                _mm512_store_si512(ix + 16 * h, _mm512_or_si512(lo, _mm512_slli_epi32(hi, 8)));
            }
            memcpy(sm, c + 32, 32);
            const uint8_t* sc = c + 72;
            for (int m = 0; m < 4; m++) {
                const __m512i g = _mm512_i32gather_epi64(_mm256_load_si256((const __m256i*)(ix + 8 * m)), (const long long*)gg_iq2s_grid, 8);
                const __m512i dt = gl::sdot(_mm512_setzero_si512(), g, _mm512_loadu_si512(a + 64 * m), sm[m]);
                const float da = (m < 2 ? da0 : da1) * 0.125f;
                const uint8_t s0 = sc[2 * m], s1 = sc[2 * m + 1];
                const __m512 fv = _mm512_permutexvar_ps(
                    _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
                    _mm512_castps128_ps512(_mm_setr_ps((float)(2 * (s0 & 15) + 1) * da, (float)(2 * (s0 >> 4) + 1) * da,
                                                       (float)(2 * (s1 & 15) + 1) * da, (float)(2 * (s1 >> 4) + 1) * da)));
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dt), fv, acc);
            }
        } else {   // XF_IQ3S: 9-bit indices qs[i] | ((qh[i / 8] >> (i % 8)) & 1) << 8, i = 0..63
            memcpy(sm, c + 72, 32);
            const uint8_t* sc = c + 104;
            const __m512i bit = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            for (int m = 0; m < 4; m++) {
                uint16_t hw;
                memcpy(&hw, c + 64 + 2 * m, 2);
                const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(hw), bit), _mm512_set1_epi32(1));
                const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + 16 * m))), _mm512_slli_epi32(hi, 8));
                const __m512i g = _mm512_i32gather_epi32(idx, (const int*)gg_iq3s_grid, 4);
                const __m512i dt = gl::sdot(_mm512_setzero_si512(), g, _mm512_loadu_si512(a + 64 * m), sm[m]);
                const float da = m < 2 ? da0 : da1;
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dt),
                                      gl::two((float)(1 + 2 * (sc[m] & 15)) * da, (float)(1 + 2 * (sc[m] >> 4)) * da), acc);
            }
        }
    }
    return _mm512_reduce_add_ps(acc);
}

// ---------------------------------------------------------------- plane kernels (one row; ap = plane-order activations)

// IQ2_XXS, two blocks per step (nb even)
template <bool NAT = false>
static inline float gg_row_dot_p2(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                   const uint8_t* pf = nullptr) {   // pf: a later row's bytes, prefetched along ([dsv4-dec1-ggpf])
    constexpr int CS = NAT ? 66 : 64, DS = NAT ? 33 : 1;
    const gl::Planes& P = gl::planes();
    alignas(64) static const uint8_t pidx[64] = {
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
        64, 65, 66, 67, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123};
    const __m512i perm = _mm512_load_si512(pidx);
    const __m512i ctrl = _mm512_set1_epi64((long long)(32ull | (39ull << 8) | (46ull << 16) | (53ull << 24)));
    const __m512i sperm = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);   // upper dwords
    __m512 acc = _mm512_setzero_ps();
    for (int p = 0; p < nb / 2; p++, c += 2 * CS, ap += 512) {
        if (pf) {
            _mm_prefetch((const char*)(pf + 2 * CS * p), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + 2 * CS * p + 64), _MM_HINT_T0);
            if (NAT) _mm_prefetch((const char*)(pf + 2 * CS * p + 128), _MM_HINT_T0);
        }
        const __m512i r0 = _mm512_loadu_si512(c), r1 = _mm512_loadu_si512(c + CS);
        const __m512i idx = _mm512_permutex2var_epi8(r0, perm, r1);
        const __mmask64 top = _mm512_movepi8_mask(idx);
        const __m512i sg = gl::parity7(_mm512_permutex2var_epi8(_mm512_multishift_epi64_epi8(ctrl, r0), perm,
                                                                _mm512_multishift_epi64_epi8(ctrl, r1)));
        const __m512i hi = _mm512_permutex2var_epi32(r0, sperm, r1);   // scale in bits 28..31
        const __m512 sc = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(_mm512_srli_epi32(hi, 28), 1), _mm512_set1_epi32(1)));
        const float d0 = gl::h2f(ds[2 * p * DS]) * 0.125f, d1 = gl::h2f(ds[(2 * p + 1) * DS]) * 0.125f;
        const float f0 = d0 * ascl[4 * p], f1 = d0 * ascl[4 * p + 1], f2 = d1 * ascl[4 * p + 2], f3 = d1 * ascl[4 * p + 3];
        const __m512 f = _mm512_mul_ps(sc, _mm512_setr_ps(f0, f0, f0, f0, f1, f1, f1, f1, f2, f2, f2, f2, f3, f3, f3, f3));
        __m512i dot = _mm512_setzero_si512();
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++)
            dot = gl::sdot(dot, gl::plane_lookup(P.p2[j], idx, top), _mm512_load_si512(ap + 64 * j),
                           _mm512_test_epi8_mask(sg, _mm512_set1_epi8((char)(1 << j))));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), f, acc);
    }
    return _mm512_reduce_add_ps(acc);
}
// IQ3_XXS, one block per step
template <bool NAT = false>
static inline float gg_row_dot_p3(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                   const uint8_t* pf = nullptr) {
    constexpr int CS = NAT ? 98 : 96, DS = NAT ? 49 : 1;
    const gl::Planes& P = gl::planes();
    const __m512i ctrl = _mm512_set1_epi64((long long)(0ull | (0ull << 8) | (7ull << 16) | (7ull << 24) | (14ull << 32) |
                                                       (14ull << 40) | (21ull << 48) | (21ull << 56)));
    const __m512i dsel = _mm512_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);   // dword lane k -> sub-block k / 2
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS, ap += 256) {
        if (pf) {
            _mm_prefetch((const char*)(pf + CS * b), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + CS * b + 64), _MM_HINT_T0);
        }
        const __m512i idx = _mm512_loadu_si512(c);
        const __mmask64 top = _mm512_movepi8_mask(idx);
        const __m512i sw = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)(c + 64)));
        __m512i sg = gl::parity7(_mm512_multishift_epi64_epi8(ctrl, sw));
        sg = _mm512_mask_blend_epi8((__mmask64)0xAAAAAAAAAAAAAAAAull, sg, _mm512_and_si512(_mm512_srli_epi16(sg, 4), _mm512_set1_epi8(0x0f)));
        const __m512i sc = _mm512_srli_epi32(_mm512_permutexvar_epi32(dsel, sw), 28);
        const float d = gl::h2f(ds[b * DS]) * 0.25f, da0 = d * ascl[2 * b], da1 = d * ascl[2 * b + 1];
        const __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(sc, 1), _mm512_set1_epi32(1))), gl::two(da0, da1));
        __m512i dot = _mm512_setzero_si512();
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++)
            dot = gl::sdot(dot, gl::plane_lookup(P.p3[j], idx, top), _mm512_load_si512(ap + 64 * j),
                           _mm512_test_epi8_mask(sg, _mm512_set1_epi8((char)(1 << j))));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), f, acc);
    }
    return _mm512_reduce_add_ps(acc);
}
// IQ3_S, one block per step: 9-bit index = low byte (planes of 256 per half) + qh bit (64-bit mask, entry i = bit i)
template <bool NAT = false>
static inline float gg_row_dot_p3s(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* ap, const float* ascl,
                                    const uint8_t* pf = nullptr) {
    constexpr int CS = NAT ? 110 : 108, DS = NAT ? 55 : 1;
    const gl::Planes& P = gl::planes();
    alignas(64) static const uint8_t sdup[64] = {   // lane i <- sign byte i / 2
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
        16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31};
    const __m512i dup = _mm512_load_si512(sdup);
    // dword lane k (entries 4k .. 4k + 3) -> sub-block k / 2 -> nibble of scales[k / 4] at 4 ((k / 2) & 1)
    const __m512i nsh = _mm512_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS, ap += 256) {
        if (pf) {
            _mm_prefetch((const char*)(pf + CS * b), _MM_HINT_T0);
            _mm_prefetch((const char*)(pf + CS * b + 64), _MM_HINT_T0);
        }
        const __m512i idx = _mm512_loadu_si512(c);
        const __mmask64 top = _mm512_movepi8_mask(idx);
        uint64_t qh;
        memcpy(&qh, c + 64, 8);
        __m512i sg = _mm512_permutexvar_epi8(dup, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(c + 72))));
        sg = _mm512_mask_blend_epi8((__mmask64)0xAAAAAAAAAAAAAAAAull, sg, _mm512_srli_epi16(sg, 4));   // odd lanes: high nibble
        uint32_t s4;
        memcpy(&s4, c + 104, 4);
        const __m512i nib = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32((int)s4), nsh), _mm512_set1_epi32(15));
        const float d = gl::h2f(ds[b * DS]);
        const __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(nib, 1), _mm512_set1_epi32(1))),
                                       gl::two(d * ascl[2 * b], d * ascl[2 * b + 1]));
        __m512i dot = _mm512_setzero_si512();
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            const __m512i lo = gl::plane_lookup(P.p3s[j], idx, top), hi = gl::plane_lookup(P.p3s[j] + 256, idx, top);
            dot = gl::sdot(dot, _mm512_mask_blend_epi8((__mmask64)qh, lo, hi), _mm512_load_si512(ap + 64 * j),
                           _mm512_test_epi8_mask(sg, _mm512_set1_epi8((char)(1 << j))));
        }
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), f, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

// [dsv4-dec1-lmr] NR activation rows against one weight row: the weight decode (plane lookups, sign masks, scales)
// once, the signed dot per activation row; each row's float sequence (int dot -> cvt -> fmadd with its own scale,
// reduce) is the one-row kernel's: bit-identical per row. ap[i] / ascl[i] / out[i] per activation row
template <int NR, bool NAT = false>
static inline void gg_rows_dot_p2(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* const* ap,
                                  const float* const* ascl, float* out) {
    constexpr int CS = NAT ? 66 : 64, DS = NAT ? 33 : 1;
    const gl::Planes& P = gl::planes();
    alignas(64) static const uint8_t pidx[64] = {
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
        64, 65, 66, 67, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123};
    const __m512i perm = _mm512_load_si512(pidx);
    const __m512i ctrl = _mm512_set1_epi64((long long)(32ull | (39ull << 8) | (46ull << 16) | (53ull << 24)));
    const __m512i sperm = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    __m512 acc[NR];
    for (int i = 0; i < NR; i++) acc[i] = _mm512_setzero_ps();
    for (int p = 0; p < nb / 2; p++, c += 2 * CS) {
        const __m512i r0 = _mm512_loadu_si512(c), r1 = _mm512_loadu_si512(c + CS);
        const __m512i idx = _mm512_permutex2var_epi8(r0, perm, r1);
        const __mmask64 top = _mm512_movepi8_mask(idx);
        const __m512i sg = gl::parity7(_mm512_permutex2var_epi8(_mm512_multishift_epi64_epi8(ctrl, r0), perm,
                                                                _mm512_multishift_epi64_epi8(ctrl, r1)));
        const __m512i hi = _mm512_permutex2var_epi32(r0, sperm, r1);
        const __m512 sc = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(_mm512_srli_epi32(hi, 28), 1), _mm512_set1_epi32(1)));
        const float d0 = gl::h2f(ds[2 * p * DS]) * 0.125f, d1 = gl::h2f(ds[(2 * p + 1) * DS]) * 0.125f;
        __m512i g[8];
        __mmask64 m[8];
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++) {
            g[j] = gl::plane_lookup(P.p2[j], idx, top);
            m[j] = _mm512_test_epi8_mask(sg, _mm512_set1_epi8((char)(1 << j)));
        }
#pragma GCC unroll 6
        for (int i = 0; i < NR; i++) {
            const float* as = ascl[i];
            const float f0 = d0 * as[4 * p], f1 = d0 * as[4 * p + 1], f2 = d1 * as[4 * p + 2], f3 = d1 * as[4 * p + 3];
            const __m512 f = _mm512_mul_ps(sc, _mm512_setr_ps(f0, f0, f0, f0, f1, f1, f1, f1, f2, f2, f2, f2, f3, f3, f3, f3));
            const int8_t* a = ap[i] + 512 * p;
            __m512i dot = _mm512_setzero_si512();
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++) dot = gl::sdot(dot, g[j], _mm512_load_si512(a + 64 * j), m[j]);
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), f, acc[i]);
        }
    }
    for (int i = 0; i < NR; i++) out[i] = _mm512_reduce_add_ps(acc[i]);
}
template <int NR, bool NAT = false>
static inline void gg_rows_dot_p3(const uint8_t* c, const uint16_t* ds, int nb, const int8_t* const* ap,
                                  const float* const* ascl, float* out) {
    constexpr int CS = NAT ? 98 : 96, DS = NAT ? 49 : 1;
    const gl::Planes& P = gl::planes();
    const __m512i ctrl = _mm512_set1_epi64((long long)(0ull | (0ull << 8) | (7ull << 16) | (7ull << 24) | (14ull << 32) |
                                                       (14ull << 40) | (21ull << 48) | (21ull << 56)));
    const __m512i dsel = _mm512_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    __m512 acc[NR];
    for (int i = 0; i < NR; i++) acc[i] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++, c += CS) {
        const __m512i idx = _mm512_loadu_si512(c);
        const __mmask64 top = _mm512_movepi8_mask(idx);
        const __m512i sw = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)(c + 64)));
        __m512i sg = gl::parity7(_mm512_multishift_epi64_epi8(ctrl, sw));
        sg = _mm512_mask_blend_epi8((__mmask64)0xAAAAAAAAAAAAAAAAull, sg, _mm512_and_si512(_mm512_srli_epi16(sg, 4), _mm512_set1_epi8(0x0f)));
        const __m512i sc = _mm512_srli_epi32(_mm512_permutexvar_epi32(dsel, sw), 28);
        const __m512 scf = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(sc, 1), _mm512_set1_epi32(1)));
        const float d = gl::h2f(ds[b * DS]) * 0.25f;
        __m512i g[4];
        __mmask64 m[4];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            g[j] = gl::plane_lookup(P.p3[j], idx, top);
            m[j] = _mm512_test_epi8_mask(sg, _mm512_set1_epi8((char)(1 << j)));
        }
#pragma GCC unroll 6
        for (int i = 0; i < NR; i++) {
            const float da0 = d * ascl[i][2 * b], da1 = d * ascl[i][2 * b + 1];
            const __m512 f = _mm512_mul_ps(scf, gl::two(da0, da1));
            const int8_t* a = ap[i] + 256 * b;
            __m512i dot = _mm512_setzero_si512();
#pragma GCC unroll 4
            for (int j = 0; j < 4; j++) dot = gl::sdot(dot, g[j], _mm512_load_si512(a + 64 * j), m[j]);
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), f, acc[i]);
        }
    }
    for (int i = 0; i < NR; i++) out[i] = _mm512_reduce_add_ps(acc[i]);
}

// ---------------------------------------------------------------- matrix rows

// [dsv4-dec1-ggpf] software prefetch distance in rows for the plane kernels (loads only: outputs unchanged); the host
// expert copies sit on 4 KB pages where the hardware streamers stop at each page (as R13's FP4 kernel).
// DSV4_GGPFD=<rows> (0 = off); default set from the lane bench
static inline int gg_pfd() {
    static const int v = getenv("DSV4_GGPFD") ? atoi(getenv("DSV4_GGPFD")) : 0;
    return v;
}
// rows [r0, r1) of a matrix [N][K] in engine layout (q codes, s fp16 d) -> y[r]. a = plain int8 activations,
// ap = the same in plane order (gg_perm_act with gg_plane_kind(mf, K)) or nullptr (then permuted here: tests)
static inline void k_ggv_any(int mf, const uint8_t* q, const uint16_t* s, int K, int r0, int r1, const int8_t* a,
                             const float* ascl, float* y, const int8_t* ap = nullptr) {
    const int nb = K / 256, cb = gg_cbytes(mf);
    const int kind = gg_plane_kind(mf, K);
    if (kind) {
        alignas(64) int8_t loc[GG_PERM_MAXK];
        if (!ap) {
            gg_perm_act(kind, a, K, loc);
            ap = loc;
        }
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

// [dsv4-dec1-lmr] rows [r0, r1) of one matrix against n (1..6) activation rows that share it (a lane group: one
// expert, several rows / jobs) -> y[i][r]; the weights are decoded once per row for all n (bit-identical per output
// to k_ggv_any / k_ggv_nat). Only the IQ2_XXS / IQ3_XXS plane paths; false = not handled (caller: one row at a time).
// nat: raw = the GGUF matrix's first block (row r at r x nb x (cbytes + 2), d first); else q / s engine layout
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

// [dsv4-rel-ram]: rows [r0, r1) of a GGUF matrix in its own layout (raw = the matrix's first block; row r at
// r x (K / 256) x block bytes) -> y[r]; for experts that have no host copy (read through the file's page cache)
static inline void k_ggv_nat(int mf, const uint8_t* raw, int K, int r0, int r1, const int8_t* a, const float* ascl,
                             float* y, const int8_t* ap = nullptr) {
    const int nb = K / 256, bb = gg_cbytes(mf) + 2;
    const int kind = gg_plane_kind(mf, K);
    if (kind) {
        alignas(64) int8_t loc[GG_PERM_MAXK];
        if (!ap) {
            gg_perm_act(kind, a, K, loc);
            ap = loc;
        }
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
