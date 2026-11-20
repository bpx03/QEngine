// GLM CPU lane kernels (see glm_cpu.h / glm_cpu.cpp). [glm-rel-isa] compiled twice: GLM_ISA_AVX512=1 (the Zen 4 set,
// the kernels exactly as before) and GLM_ISA_AVX512=0 (x86-64-v3: avx2, fma, f16c, bmi2). Pattern of [dsv4-rel3-isa]
// (dsv4_cpu_isa.cpp): every header with inline (shared) functions is included BEFORE the target pragma, so their
// out-of-line copies are plain x86-64 code in both objects; what follows the pragma is static or has a per-build name
// (CpuLane::work_avx512 / _avx2, pf_work_avx512 / _avx2). The threads, pinning and job plumbing are in glm_cpu.cpp
// (plain x86-64, compiled once), which picks the build at lane start (lane_isa()).
// AVX2 = the same arithmetic per output column: every op of the kernels is vertical (per column), so the AVX2 kernels
// walk the 32-column tile as four 8-column quarters (16 ymm registers) with each column's op sequence unchanged:
// integer dot sums are exact (u8 x s8 via vpmaddubsw + vpmaddwd: weights 0..15, activations -127..127, so a pair sum
// is at most 2 * 15 * 127 = 3810 and never saturates int16), the float epilogue (cvt, fma, mul, fma into the
// accumulator per group in order) is the same instruction sequence on 8 lanes instead of 16 -> bit-identical results.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <immintrin.h>

#include "glm_cpu.h"
#include "../common.h"

#ifndef GLM_ISA_AVX512
#error "build with -DGLM_ISA_AVX512=1 (AVX-512 kernels) or =0 (AVX2 kernels)"
#endif
#if !defined(__GNUC__) || defined(__clang__)
#error "GCC only (the target pragmas; the AVX-512 build must match the former -march=znver4 code)"
#endif
// ISA lists, no arch= / tune= (GCC 13 refuses to inline the plain helpers into functions of another arch / tune, DSV4
// review rel3 H1); tuning comes from -mtune=znver4 on the command line. The lists = lane_isa()'s checks (glm_cpu.cpp)
#if GLM_ISA_AVX512
#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512ifma,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,avx512bf16,gfni,vaes,vpclmulqdq,avx2,avx,fma,f16c,bmi,bmi2,lzcnt,movbe,popcnt")
#define ISA_FN(n) n##_avx512
#else
#pragma GCC target("avx2,avx,fma,f16c,bmi,bmi2,lzcnt,movbe,popcnt")
#define ISA_FN(n) n##_avx2
#endif

namespace glm {

static int g_pf = getenv("GLM_CPU_PF") ? atoi(getenv("GLM_CPU_PF")) : 0;
static float g_dsp = getenv("GLM_DSP") ? (float)atof(getenv("GLM_DSP")) : 0.f;  // LOSSY opt-in

// x[K] fp32 -> int8 per G-group (G = 128 or 64): xe / xo = packed bytes (x[8i], x[8i+2], x[8i+4], x[8i+6]) / odd
// ones, scale, sum
template <int G = 128>
static void quant_grp(const float* x, int g, int32_t* xe, int32_t* xo, float* xs, int32_t* sx, int32_t* re, int32_t* ro,
                      float* rs, int32_t* rsx) {
    float m = 0.f;
    for (int i = 0; i < G; i++) m = std::max(m, std::fabs(x[g * G + i]));
    const float s = m > 0 ? m / 127.f : 1.f, inv = 1.f / s;
    int sum = 0;
    int8_t q[G];
    for (int i = 0; i < G; i++) {
        int v = (int)lrintf(x[g * G + i] * inv);
        v = std::min(127, std::max(-127, v));
        q[i] = (int8_t)v;
        sum += v;
    }
    xs[g] = s;
    sx[g] = sum;
    // residual x - q s in [-s/2, s/2] -> second int8 with scale s / 254
    const float s2 = s / 254.f, inv2 = 1.f / s2;
    int sum2 = 0;
    int8_t q2[G];
    for (int i = 0; i < G; i++) {
        int v = (int)lrintf((x[g * G + i] - q[i] * s) * inv2);
        v = std::min(127, std::max(-127, v));
        q2[i] = (int8_t)v;
        sum2 += v;
    }
    rs[g] = s2;
    rsx[g] = sum2;
    for (int j = 0; j < G / 8; j++) {
        const int8_t* p = q + j * 8;
        const int8_t* p2 = q2 + j * 8;
        uint8_t e[4] = {(uint8_t)p[0], (uint8_t)p[2], (uint8_t)p[4], (uint8_t)p[6]};
        uint8_t o[4] = {(uint8_t)p[1], (uint8_t)p[3], (uint8_t)p[5], (uint8_t)p[7]};
        uint8_t e2[4] = {(uint8_t)p2[0], (uint8_t)p2[2], (uint8_t)p2[4], (uint8_t)p2[6]};
        uint8_t o2[4] = {(uint8_t)p2[1], (uint8_t)p2[3], (uint8_t)p2[5], (uint8_t)p2[7]};
        memcpy(&xe[g * (G / 8) + j], e, 4);
        memcpy(&xo[g * (G / 8) + j], o, 4);
        memcpy(&re[g * (G / 8) + j], e2, 4);
        memcpy(&ro[g * (G / 8) + j], o2, 4);
    }
}

struct Act {
    const int32_t *xe, *xo, *sx, *re, *ro, *rsx;
    const float *xs, *rs;
};

// 16 fp32 columns of a tile: one zmm (AVX-512) or two ymm (AVX2); the plumbing below uses only these helpers
#if GLM_ISA_AVX512
using VF = __m512;
static inline VF vf_zero() { return _mm512_setzero_ps(); }
static inline VF vf_add(VF a, VF b) { return _mm512_add_ps(a, b); }
static inline VF vf_loadu(const float* p) { return _mm512_loadu_ps(p); }
static inline void vf_store(float* p, VF v) { _mm512_store_ps(p, v); }
static inline void vf_storeu(float* p, VF v) { _mm512_storeu_ps(p, v); }
#else
struct VF { __m256 h[2]; };
static inline VF vf_zero() { return VF{{_mm256_setzero_ps(), _mm256_setzero_ps()}}; }
static inline VF vf_add(VF a, VF b) { return VF{{_mm256_add_ps(a.h[0], b.h[0]), _mm256_add_ps(a.h[1], b.h[1])}}; }
static inline VF vf_loadu(const float* p) { return VF{{_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}}; }
static inline void vf_store(float* p, VF v) { _mm256_store_ps(p, v.h[0]); _mm256_store_ps(p + 8, v.h[1]); }
static inline void vf_storeu(float* p, VF v) { _mm256_storeu_ps(p, v.h[0]); _mm256_storeu_ps(p + 8, v.h[1]); }
#endif

#if GLM_ISA_AVX512
// 32 columns [c0, c0 + 32) of y_i = x_i . W for NR activation rows (weights read once), W: qweight q [K/8][ld],
// scales s [K/G][ld] fp16 (G = 8 GR: 128 for gate/up, 64 for down); f0 / f1 [NR] accumulate
template <int NR, int GR>
static inline void tileM(const int32_t* q, const uint16_t* s, int ld, int c0, int K, const Act* A, __m512* f0,
                         __m512* f1, const uint8_t* sk = nullptr) {
    const __m512i m4 = _mm512_set1_epi32(0x0F0F0F0F);
    for (int g = 0; g < K / (8 * GR); g++) {
        __m512i a0[NR], a1[NR], b0[NR], b1[NR];
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) a0[i] = a1[i] = b0[i] = b1[i] = _mm512_setzero_si512();
        const int32_t* qr = q + (size_t)g * GR * ld + c0;
#pragma GCC unroll 2
        for (int r = 0; r < GR; r++) {
            if (g_pf) {  // software prefetch g_pf rows ahead (rows are ld * 4 bytes apart: strided stream)
                const int32_t* pf = q + ((size_t)g * GR + r + g_pf) * ld + c0;
                if ((size_t)g * GR + r + g_pf < (size_t)K / 8) { _mm_prefetch((const char*)pf, _MM_HINT_T0); _mm_prefetch((const char*)(pf + 16), _MM_HINT_T0); }
            }
            if (sk && sk[g * GR + r]) continue;  // skipped down block (its h is zero: no contribution)
            const __m512i w0 = _mm512_loadu_si512(qr + (size_t)r * ld);
            const __m512i w1 = _mm512_loadu_si512(qr + (size_t)r * ld + 16);
            const __m512i l0 = _mm512_and_si512(w0, m4), h0 = _mm512_and_si512(_mm512_srli_epi32(w0, 4), m4);
            const __m512i l1 = _mm512_and_si512(w1, m4), h1 = _mm512_and_si512(_mm512_srli_epi32(w1, 4), m4);
#pragma GCC unroll 4
            for (int i = 0; i < NR; i++) {
                const int ix = g * GR + r;
                const __m512i be = _mm512_set1_epi32(A[i].xe[ix]), bo = _mm512_set1_epi32(A[i].xo[ix]);
                const __m512i ce = _mm512_set1_epi32(A[i].re[ix]), co = _mm512_set1_epi32(A[i].ro[ix]);
                a0[i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(a0[i], l0, be), h0, bo);
                a1[i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(a1[i], l1, be), h1, bo);
                b0[i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(b0[i], l0, ce), h0, co);
                b1[i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(b1[i], l1, ce), h1, co);
            }
        }
        const uint16_t* sr = s + (size_t)g * ld + c0;
        const __m512 s0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)sr));
        const __m512 s1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(sr + 16)));
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) {
            const __m512i corr = _mm512_set1_epi32(8 * A[i].sx[g]), corr2 = _mm512_set1_epi32(8 * A[i].rsx[g]);
            const __m512 xsv = _mm512_set1_ps(A[i].xs[g]), rsv = _mm512_set1_ps(A[i].rs[g]);
            const __m512 v0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0[i], corr)), xsv,
                                              _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(b0[i], corr2)), rsv));
            const __m512 v1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1[i], corr)), xsv,
                                              _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(b1[i], corr2)), rsv));
            f0[i] = _mm512_fmadd_ps(v0, s0, f0[i]);
            f1[i] = _mm512_fmadd_ps(v1, s1, f1[i]);
        }
    }
}
// INT3 gate/up tile (SliceL::gu3): q = [128][3][32] i32 planes of one 32-column tile, s = [32][32] f16; same activation
// packing as the INT4 kernel (set j = 2t + p of a 32-row block = xe / xo of k8 row 4kb + t), zero point 4
template <int NR>
static inline void tileM3(const int32_t* q, const uint16_t* s, const Act* A, __m512* f0, __m512* f1) {
    const __m512i m2 = _mm512_set1_epi32(0x03030303), m1 = _mm512_set1_epi32(0x01010101);
    for (int g = 0; g < H / 128; g++) {
        __m512i a0[NR], a1[NR], b0[NR], b1[NR];
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) a0[i] = a1[i] = b0[i] = b1[i] = _mm512_setzero_si512();
        for (int kb = 4 * g; kb < 4 * g + 4; kb++) {
            const int32_t* qr = q + (size_t)kb * 96;
            const __m512i lo0 = _mm512_loadu_si512(qr), lo1 = _mm512_loadu_si512(qr + 16);
            const __m512i hi0 = _mm512_loadu_si512(qr + 32), hi1 = _mm512_loadu_si512(qr + 48);
            const __m512i hb0 = _mm512_loadu_si512(qr + 64), hb1 = _mm512_loadu_si512(qr + 80);
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++) {
                const int t = j >> 1, p = j & 1, sh = 2 * (2 * (t & 1) + p);
                const __m512i s0 = t < 2 ? lo0 : hi0, s1 = t < 2 ? lo1 : hi1;
                const __m512i v0 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi32(s0, sh), m2),
                                                   _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hb0, j), m1), 2));
                const __m512i v1 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi32(s1, sh), m2),
                                                   _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hb1, j), m1), 2));
                const int ix = 4 * kb + t;
#pragma GCC unroll 4
                for (int i = 0; i < NR; i++) {
                    const __m512i bx = _mm512_set1_epi32(p ? A[i].xo[ix] : A[i].xe[ix]);
                    const __m512i cx = _mm512_set1_epi32(p ? A[i].ro[ix] : A[i].re[ix]);
                    a0[i] = _mm512_dpbusd_epi32(a0[i], v0, bx);
                    a1[i] = _mm512_dpbusd_epi32(a1[i], v1, bx);
                    b0[i] = _mm512_dpbusd_epi32(b0[i], v0, cx);
                    b1[i] = _mm512_dpbusd_epi32(b1[i], v1, cx);
                }
            }
        }
        const uint16_t* sr = s + (size_t)g * 32;
        const __m512 sc0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)sr));
        const __m512 sc1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(sr + 16)));
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) {
            const __m512i corr = _mm512_set1_epi32(4 * A[i].sx[g]), corr2 = _mm512_set1_epi32(4 * A[i].rsx[g]);
            const __m512 xsv = _mm512_set1_ps(A[i].xs[g]), rsv = _mm512_set1_ps(A[i].rs[g]);
            const __m512 v0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0[i], corr)), xsv,
                                              _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(b0[i], corr2)), rsv));
            const __m512 v1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1[i], corr)), xsv,
                                              _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(b1[i], corr2)), rsv));
            f0[i] = _mm512_fmadd_ps(v0, sc0, f0[i]);
            f1[i] = _mm512_fmadd_ps(v1, sc1, f1[i]);
        }
    }
}
#else
// AVX2 tiles: quarter qt = columns [8 qt, 8 qt + 8) of the 32-column tile = f0[i].h[0], f0[i].h[1], f1[i].h[0],
// f1[i].h[1]. Per column the same integer sums (exact) and the same float sequence per group as the AVX-512 tile.
// vpdpbusd(acc, u8 w, s8 a) per i32 lane = acc + sum of 4 products: vpmaddubsw (pairs, |sum| <= 2 * 15 * 127, no
// saturation) + vpmaddwd by 1 (exact i32) + add
static inline __m256i dp8(__m256i acc, __m256i w, __m256i a) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1)));
}
// two products (low and high nibbles): the int16 pair sums add before the widening (|sum| <= 4 * 15 * 127 = 7620)
static inline __m256i dp8x2(__m256i acc, __m256i w0, __m256i a0, __m256i w1, __m256i a1) {
    const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(w0, a0), _mm256_maddubs_epi16(w1, a1));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
static inline __m256& vq(VF* f0, VF* f1, int i, int qt) { return qt < 2 ? f0[i].h[qt] : f1[i].h[qt - 2]; }
template <int NR, int GR>
static inline void tileM(const int32_t* q, const uint16_t* s, int ld, int c0, int K, const Act* A, VF* f0, VF* f1,
                         const uint8_t* sk = nullptr) {
    const __m256i m4 = _mm256_set1_epi32(0x0F0F0F0F);
    for (int qt = 0; qt < 4; qt++) {
        __m256 f[NR];
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) f[i] = vq(f0, f1, i, qt);
        for (int g = 0; g < K / (8 * GR); g++) {
            __m256i a[NR], b[NR];
#pragma GCC unroll 4
            for (int i = 0; i < NR; i++) a[i] = b[i] = _mm256_setzero_si256();
            const int32_t* qr = q + (size_t)g * GR * ld + c0 + 8 * qt;
#pragma GCC unroll 2
            for (int r = 0; r < GR; r++) {
                if (g_pf && qt == 0) {  // software prefetch g_pf rows ahead (the whole 128-byte tile row, once)
                    const int32_t* pf = q + ((size_t)g * GR + r + g_pf) * ld + c0;
                    if ((size_t)g * GR + r + g_pf < (size_t)K / 8) { _mm_prefetch((const char*)pf, _MM_HINT_T0); _mm_prefetch((const char*)(pf + 16), _MM_HINT_T0); }
                }
                if (sk && sk[g * GR + r]) continue;  // skipped down block (its h is zero: no contribution)
                const __m256i w = _mm256_loadu_si256((const __m256i*)(qr + (size_t)r * ld));
                const __m256i l = _mm256_and_si256(w, m4), h = _mm256_and_si256(_mm256_srli_epi32(w, 4), m4);
#pragma GCC unroll 4
                for (int i = 0; i < NR; i++) {
                    const int ix = g * GR + r;
                    a[i] = dp8x2(a[i], l, _mm256_set1_epi32(A[i].xe[ix]), h, _mm256_set1_epi32(A[i].xo[ix]));
                    b[i] = dp8x2(b[i], l, _mm256_set1_epi32(A[i].re[ix]), h, _mm256_set1_epi32(A[i].ro[ix]));
                }
            }
            const __m256 sv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + (size_t)g * ld + c0 + 8 * qt)));
#pragma GCC unroll 4
            for (int i = 0; i < NR; i++) {
                const __m256i corr = _mm256_set1_epi32(8 * A[i].sx[g]), corr2 = _mm256_set1_epi32(8 * A[i].rsx[g]);
                const __m256 xsv = _mm256_set1_ps(A[i].xs[g]), rsv = _mm256_set1_ps(A[i].rs[g]);
                const __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a[i], corr)), xsv,
                                                 _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(b[i], corr2)), rsv));
                f[i] = _mm256_fmadd_ps(v, sv, f[i]);
            }
        }
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) vq(f0, f1, i, qt) = f[i];
    }
}
template <int NR>
static inline void tileM3(const int32_t* q, const uint16_t* s, const Act* A, VF* f0, VF* f1) {
    const __m256i m2 = _mm256_set1_epi32(0x03030303), m1 = _mm256_set1_epi32(0x01010101);
    for (int qt = 0; qt < 4; qt++) {
        __m256 f[NR];
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) f[i] = vq(f0, f1, i, qt);
        for (int g = 0; g < H / 128; g++) {
            __m256i a[NR], b[NR];
#pragma GCC unroll 4
            for (int i = 0; i < NR; i++) a[i] = b[i] = _mm256_setzero_si256();
            for (int kb = 4 * g; kb < 4 * g + 4; kb++) {
                const int32_t* qr = q + (size_t)kb * 96 + 8 * qt;
                const __m256i lo = _mm256_loadu_si256((const __m256i*)qr), hi = _mm256_loadu_si256((const __m256i*)(qr + 32));
                const __m256i hb = _mm256_loadu_si256((const __m256i*)(qr + 64));
#pragma GCC unroll 8
                for (int j = 0; j < 8; j++) {
                    const int t = j >> 1, p = j & 1, sh = 2 * (2 * (t & 1) + p);
                    const __m256i s0 = t < 2 ? lo : hi;
                    const __m256i v = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(s0, sh), m2),
                                                      _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, j), m1), 2));
                    const int ix = 4 * kb + t;
#pragma GCC unroll 4
                    for (int i = 0; i < NR; i++) {
                        a[i] = dp8(a[i], v, _mm256_set1_epi32(p ? A[i].xo[ix] : A[i].xe[ix]));
                        b[i] = dp8(b[i], v, _mm256_set1_epi32(p ? A[i].ro[ix] : A[i].re[ix]));
                    }
                }
            }
            const __m256 sv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + (size_t)g * 32 + 8 * qt)));
#pragma GCC unroll 4
            for (int i = 0; i < NR; i++) {
                const __m256i corr = _mm256_set1_epi32(4 * A[i].sx[g]), corr2 = _mm256_set1_epi32(4 * A[i].rsx[g]);
                const __m256 xsv = _mm256_set1_ps(A[i].xs[g]), rsv = _mm256_set1_ps(A[i].rs[g]);
                const __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a[i], corr)), xsv,
                                                 _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(b[i], corr2)), rsv));
                f[i] = _mm256_fmadd_ps(v, sv, f[i]);
            }
        }
#pragma GCC unroll 4
        for (int i = 0; i < NR; i++) vq(f0, f1, i, qt) = f[i];
    }
}
#endif
static inline void tile3_rows(int nr, const int32_t* q, const uint16_t* s, const Act* A, VF* f0, VF* f1) {
    switch (nr) {
        case 1: tileM3<1>(q, s, A, f0, f1); break;
        case 2: tileM3<2>(q, s, A, f0, f1); break;
        case 3: tileM3<3>(q, s, A, f0, f1); break;
        default: tileM3<4>(q, s, A, f0, f1); break;
    }
}
template <int GR>
static inline void tile_rows(int nr, const int32_t* q, const uint16_t* s, int ld, int c0, int K, const Act* A,
                             VF* f0, VF* f1, const uint8_t* sk = nullptr) {
    switch (nr) {
        case 1: tileM<1, GR>(q, s, ld, c0, K, A, f0, f1, sk); break;
        case 2: tileM<2, GR>(q, s, ld, c0, K, A, f0, f1, sk); break;
        case 3: tileM<3, GR>(q, s, ld, c0, K, A, f0, f1, sk); break;
        default: tileM<4, GR>(q, s, ld, c0, K, A, f0, f1, sk); break;
    }
}

static inline float siluf(float x) { return x / (1.f + std::exp(-x)); }

void CpuLane::ISA_FN(work)(int id) {
    // round 11 (review 2 F1a): while a thread waits at a barrier it touches its own upcoming tiles (4 KB per poll):
    // first its gate/up tiles (at bar0), then its down tiles (at bar / bar2) -> the DRAM keeps streaming
    struct Cur {
        CpuLane* L; int id; int ph = 0; int j = 0, e = 0, r = 0, part = 0; size_t off = 0; unsigned acc = 0;
        bool next(const char*& p, size_t& n) {  // the next own range; false when exhausted
            CpuLane& Q = *L;
            for (;;) {
                if (ph == 0) {  // gate / up tiles of the own static list
                    if (!Q.steal || j >= Q.lA_n[id]) { ph = 1; j = id; e = 0; r = 0; continue; }
                    const int t = Q.lA[id * Q.lA_cap + j], ex_ = t / (INTER / 32), c = (t % (INTER / 32)) * 32;
                    int rr = 0;
                    while (rr + 1 < Q.hl->ns && c >= Q.hl->e0[rr + 1]) rr++;
                    const SliceL& S = Q.hl->lay(Q.ex[ex_], rr);
                    const char* b = Q.ex[ex_] + Q.hl->off[rr];
                    const size_t tc = (size_t)(c - Q.hl->e0[rr]) / 32;
                    const size_t tb = S.gu3 ? (size_t)128 * 3 * 32 * 4 : (size_t)512 * 32 * 4;
                    p = b + (part ? S.uq : S.gq) + tc * tb; n = tb;
                    if (++part == 2) { part = 0; j++; }
                    return true;
                }
                if (ph == 1) {  // down tiles of the own output columns t = id, id + nth, ..., every expert and slice
                    if (j >= H / 32) { ph = 2; continue; }
                    const SliceL& S = Q.hl->lay(Q.ex[e], r);
                    p = Q.ex[e] + Q.hl->off[r] + S.dq + (size_t)j * (S.ni / 8) * 32 * 4; n = (size_t)(S.ni / 8) * 32 * 4;
                    if (++r == Q.hl->ns) { r = 0; if (++e == Q.ne) { e = 0; j += Q.nth; } }
                    return true;
                }
                return false;
            }
        }
        const char* p = nullptr; size_t n = 0;
        bool step() {  // touch 4 KB
            if (off >= n) { if (!next(p, n)) return false; off = 0; }
            const size_t end = std::min(n, off + 4096);
            for (; off < end; off += 64) acc += *(const volatile unsigned*)(p + off);
            return true;
        }
    } cur{this, id};
    auto bwait = [&](std::atomic<unsigned>& b, int minph) {
        b.fetch_add(1, std::memory_order_acq_rel);
        while (b.load(std::memory_order_acquire) < (unsigned)nth) {
            if (!bar_touch) { _mm_pause(); continue; }
            if (cur.ph < minph) { cur.ph = minph; cur.j = id; cur.e = 0; cur.r = 0; cur.off = cur.n = 0; }
            if (!cur.step()) _mm_pause();
        }
    };
    if (tl_on) ts[id][0] = __rdtsc();
    // phase 0: quantize the x rows (M x 32 groups over the threads)
    for (int i = id; i < nm * (H / 128); i += nth) {
        const int r = i / (H / 128), g = i % (H / 128);
        quant_grp(xin + (size_t)r * H, g, xe[r], xo[r], xs[r], sx[r], re[r], ro[r], rs[r], rsx[r]);
    }
    if (tl_on) ts[id][1] = __rdtsc();
    bwait(bar0, 0);
    if (tl_on) ts[id][2] = __rdtsc();
    // phase A: gate/up, (expert, 32 intermediate columns) tiles for the expert's rows (a tile never straddles slices)
    // static tile -> thread map keyed by the expert (the prefetch of a predicted expert lands in the computing thread's
    // caches): tile c of expert e belongs to thread (key(e) * 7 + c) % nth
    auto tileA = [&](int t) {
        const int e = t / (INTER / 32), ci = t % (INTER / 32), c = ci * 32;
        int r = 0;
        while (r + 1 < hl->ns && c >= hl->e0[r + 1]) r++;
        const SliceL& S = hl->lay(ex[e], r);
        const char* b = ex[e] + hl->off[r];
        const int c0 = c - hl->e0[r];
        const int nr = enr[e];
        Act A[MR];
        for (int i = 0; i < nr; i++) {
            const int x = erow[e][i];
            A[i] = Act{xe[x], xo[x], sx[x], re[x], ro[x], rsx[x], xs[x], rs[x]};
        }
        VF g0[MR], g1[MR], u0[MR], u1[MR];
        for (int i = 0; i < MR; i++) g0[i] = g1[i] = u0[i] = u1[i] = vf_zero();
        const size_t tq = (size_t)(c0 / 32) * 512 * 32, ts = (size_t)(c0 / 32) * 32 * 32;  // tile-major blocks
        if (S.gu3) {
            const size_t t3 = (size_t)(c0 / 32) * 128 * 3 * 32;
            tile3_rows(nr, (const int32_t*)(b + S.gq) + t3, (const uint16_t*)(b + S.gs) + ts, A, g0, g1);
            tile3_rows(nr, (const int32_t*)(b + S.uq) + t3, (const uint16_t*)(b + S.us) + ts, A, u0, u1);
        } else {
            tile_rows<16>(nr, (const int32_t*)(b + S.gq) + tq, (const uint16_t*)(b + S.gs) + ts, 32, 0, H, A, g0, g1);
            tile_rows<16>(nr, (const int32_t*)(b + S.uq) + tq, (const uint16_t*)(b + S.us) + ts, 32, 0, H, A, u0, u1);
        }
        for (int i = 0; i < nr; i++) {
            alignas(64) float gg[32], uu[32];
            vf_store(gg, g0[i]); vf_store(gg + 16, g1[i]);
            vf_store(uu, u0[i]); vf_store(uu + 16, u1[i]);
            float* hp = h.data() + ((size_t)e * MR + i) * INTER + c;
            for (int j = 0; j < 32; j++) {
                const float gv = std::min(gg[j], CLAMP), uv = std::min(std::max(uu[j], -CLAMP), CLAMP);
                hp[j] = siluf(gv) * uv * ew[e][i];
            }
        }
    };
    if (steal) {  // round 9: own tiles first (static map, prefetch locality), then the other threads' remaining ones
        for (int v = 0; v < nth; v++) {
            const int o = (id + v) % nth;
            for (;;) {
                const int j = sA[o].n.fetch_add(1, std::memory_order_relaxed);
                if (j >= lA_n[o]) break;
                tileA(lA[o * lA_cap + j]);
            }
        }
    } else
        for (int t = 0; t < ne * (INTER / 32); t++)
            if ((int)((ekey(ex[t / (INTER / 32)]) * 7 + t % (INTER / 32)) % nth) == id) tileA(t);
    if (tl_on) ts[id][3] = __rdtsc();
    bwait(bar, 1);
    if (tl_on) ts[id][4] = __rdtsc();
    if (g_dsp > 0) {  // LOSSY opt-in: per expert, the 8-row down blocks with the smallest max-over-rows |h| sums skipped
        for (int e = id; e < ne; e += nth) {
            float imp[INTER / 8], tmp[INTER / 8];
            for (int k = 0; k < INTER / 8; k++) {
                float m = 0.f;
                for (int i = 0; i < enr[e]; i++) {
                    const float* hp = h.data() + ((size_t)e * MR + i) * INTER + k * 8;
                    float a = 0.f;
                    for (int j = 0; j < 8; j++) a += std::fabs(hp[j]);
                    m = std::max(m, a);
                }
                imp[k] = tmp[k] = m;
            }
            const int nsk = std::min(INTER / 8 - 1, (int)(g_dsp * (INTER / 8)));
            std::nth_element(tmp, tmp + nsk, tmp + INTER / 8);
            const float thr = tmp[nsk];
            for (int k = 0; k < INTER / 8; k++) {
                dskip[e][k] = imp[k] < thr;
                if (dskip[e][k])
                    for (int i = 0; i < enr[e]; i++) memset(h.data() + ((size_t)e * MR + i) * INTER + k * 8, 0, 32);
            }
        }
        barrier(bar3);
    }
    // phase B: quantize h per 64-group, (expert, row, group) items over the threads
    for (int i = id; i < ne * MR * (INTER / 64); i += nth) {
        const int e = i / (MR * (INTER / 64)), j = (i / (INTER / 64)) % MR, g = i % (INTER / 64);
        if (j >= enr[e]) continue;
        const int k = e * MR + j;
        quant_grp<64>(h.data() + (size_t)k * INTER, g, he + k * (INTER / 8), ho + k * (INTER / 8), hs + k * 32,
                      hsx + k * 32, hre + k * (INTER / 8), hro + k * (INTER / 8), hrs + k * 32, hrsx + k * 32);
    }
    bwait(bar2, 1);
    if (cur.acc) pacc += cur.acc;  // r12 (review S3): no shared-line RMW per thread when the touch adds nothing
    if (tl_on) ts[id][5] = __rdtsc();
    // phase C: down, 32 output columns, all experts in order, accumulated per output row
    auto tileC = [&](int t) {
        const int c0 = t * 32;
        VF o0[MR], o1[MR];
        for (int i = 0; i < MR; i++) o0[i] = o1[i] = vf_zero();
        for (int e = 0; e < ne; e++) {
            const int nr = enr[e];
            VF f0[MR], f1[MR];
            for (int i = 0; i < MR; i++) f0[i] = f1[i] = vf_zero();
            for (int r = 0; r < hl->ns; r++) {
                const SliceL& S = hl->lay(ex[e], r);
                const char* b = ex[e] + hl->off[r];
                const int g0 = hl->e0[r] / 64, k0 = hl->e0[r] / 8;
                Act A[MR];
                for (int i = 0; i < nr; i++) {
                    const int k = e * MR + i, eo = k * (INTER / 8), eg = k * 32;
                    A[i] = Act{he + eo + k0, ho + eo + k0, hsx + eg + g0, hre + eo + k0, hro + eo + k0, hrsx + eg + g0,
                               hs + eg + g0, hrs + eg + g0};
                }
                tile_rows<8>(nr, (const int32_t*)(b + S.dq) + (size_t)t * (S.ni / 8) * 32,
                             (const uint16_t*)(b + S.ds) + (size_t)t * (S.ni / 64) * 32, 32, 0, S.ni, A, f0, f1,
                             g_dsp > 0 ? dskip[e] + k0 : nullptr);
            }
            for (int i = 0; i < nr; i++) {
                const int x = erow[e][i];
                o0[x] = vf_add(o0[x], f0[i]);
                o1[x] = vf_add(o1[x], f1[i]);
            }
        }
        for (int x = 0; x < nm; x++) {
            vf_storeu(y + (size_t)x * H + c0, o0[x]);
            vf_storeu(y + (size_t)x * H + c0 + 16, o1[x]);
        }
    };
    if (steal) {  // own output tiles t = id, id + nth, ... first, then steal (same per-tile arithmetic: bit-exact)
        for (int v = 0; v < nth; v++) {
            const int o = (id + v) % nth, no = (H / 32 - o + nth - 1) / nth;
            for (;;) {
                const int j = sC[o].n.fetch_add(1, std::memory_order_relaxed);
                if (j >= no) break;
                tileC(o + j * nth);
            }
        }
    } else
        for (int t = id; t < H / 32; t += nth) tileC(t);  // static: output tile t on thread t % nth
    if (tl_on) ts[id][6] = __rdtsc();
}

// ---------------------------------------------------------------- round 20: prefill lane ([glm-arch2-pflane])
// The decode lane's tiles and activation quantization (two-pass int8 per group), rows in chunks of MR per weight tile
// (the tile stays in L2 between chunks). Items are claimed through atomic counters; each item's arithmetic is fixed.
void CpuLane::ISA_FN(pf_work)(int id) {
    Pf& P = pf;
    constexpr int GX = H / 128, GH = INTER / 64;
    // review r20b-L6: each thread quantizes one contiguous range of groups (was every nth group: neighbouring groups'
    // outputs on one cache line written by different threads); the same arithmetic per group, so the same bits
    const int nqx = P.nx * GX;
    for (int i = (int)((long)nqx * id / nth), ie = (int)((long)nqx * (id + 1) / nth); i < ie; i++) {
        const int r = i / GX, g = i % GX;
        const size_t o8 = (size_t)r * (H / 8), og = (size_t)r * GX;
        quant_grp(P.x + (size_t)r * H, g, P.xe.data() + o8, P.xo.data() + o8, P.xs.data() + og, P.sx.data() + og,
                  P.re.data() + o8, P.ro.data() + o8, P.rs.data() + og, P.rsx.data() + og);
    }
    barrier(bar0);
    auto actx = [&](int row) {
        const size_t o8 = (size_t)row * (H / 8), og = (size_t)row * GX;
        return Act{P.xe.data() + o8, P.xo.data() + o8, P.sx.data() + og, P.re.data() + o8, P.ro.data() + o8,
                   P.rsx.data() + og, P.xs.data() + og, P.rs.data() + og};
    };
    const int nA = P.ne * (INTER / 32);
    for (;;) {
        const int t = tA.fetch_add(1, std::memory_order_relaxed);
        if (t >= nA) break;
        const int e = t / (INTER / 32), c = (t % (INTER / 32)) * 32;
        int r = 0;
        while (r + 1 < hl->ns && c >= hl->e0[r + 1]) r++;
        const SliceL& S = hl->lay(P.ex[e], r);
        const char* b = P.ex[e] + hl->off[r];
        const int c0 = c - hl->e0[r];
        const size_t tq = (size_t)(c0 / 32) * 512 * 32, ts_ = (size_t)(c0 / 32) * 32 * 32;
        for (int j0 = P.off[e]; j0 < P.off[e + 1]; j0 += MR) {
            const int nr = std::min(MR, P.off[e + 1] - j0);
            Act A[MR];
            for (int i = 0; i < nr; i++) A[i] = actx(P.rows[j0 + i]);
            VF g0[MR], g1[MR], u0[MR], u1[MR];
            for (int i = 0; i < MR; i++) g0[i] = g1[i] = u0[i] = u1[i] = vf_zero();
            if (S.gu3) {
                const size_t t3 = (size_t)(c0 / 32) * 128 * 3 * 32;
                tile3_rows(nr, (const int32_t*)(b + S.gq) + t3, (const uint16_t*)(b + S.gs) + ts_, A, g0, g1);
                tile3_rows(nr, (const int32_t*)(b + S.uq) + t3, (const uint16_t*)(b + S.us) + ts_, A, u0, u1);
            } else {
                tile_rows<16>(nr, (const int32_t*)(b + S.gq) + tq, (const uint16_t*)(b + S.gs) + ts_, 32, 0, H, A, g0, g1);
                tile_rows<16>(nr, (const int32_t*)(b + S.uq) + tq, (const uint16_t*)(b + S.us) + ts_, 32, 0, H, A, u0, u1);
            }
            for (int i = 0; i < nr; i++) {
                alignas(64) float gg[32], uu[32];
                vf_store(gg, g0[i]); vf_store(gg + 16, g1[i]);
                vf_store(uu, u0[i]); vf_store(uu + 16, u1[i]);
                float* hp = P.h.data() + (size_t)(j0 + i) * INTER + c;
                const float wv = P.w[j0 + i];
                for (int j = 0; j < 32; j++) {
                    const float gv = std::min(gg[j], CLAMP), uv = std::min(std::max(uu[j], -CLAMP), CLAMP);
                    hp[j] = siluf(gv) * uv * wv;
                }
            }
        }
    }
    barrier(bar);
    const int nqh = P.np * GH;  // (contiguous ranges, as the x groups above)
    for (int i = (int)((long)nqh * id / nth), ie = (int)((long)nqh * (id + 1) / nth); i < ie; i++) {
        const int k = i / GH, g = i % GH;
        quant_grp<64>(P.h.data() + (size_t)k * INTER, g, P.he.data() + (size_t)k * (INTER / 8), P.ho.data() + (size_t)k * (INTER / 8),
                      P.hs.data() + (size_t)k * 32, P.hsx.data() + (size_t)k * 32, P.hre.data() + (size_t)k * (INTER / 8),
                      P.hro.data() + (size_t)k * (INTER / 8), P.hrs.data() + (size_t)k * 32, P.hrsx.data() + (size_t)k * 32);
    }
    barrier(bar2);
    const int nC = P.ne * (H / 32);
    for (;;) {
        const int t = tC.fetch_add(1, std::memory_order_relaxed);
        if (t >= nC) break;
        const int e = t / (H / 32), tt = t % (H / 32), c0 = tt * 32;
        for (int j0 = P.off[e]; j0 < P.off[e + 1]; j0 += MR) {
            const int nr = std::min(MR, P.off[e + 1] - j0);
            VF f0[MR], f1[MR];
            for (int i = 0; i < MR; i++) f0[i] = f1[i] = vf_zero();
            for (int r = 0; r < hl->ns; r++) {
                const SliceL& S = hl->lay(P.ex[e], r);
                const char* b = P.ex[e] + hl->off[r];
                const int g0 = hl->e0[r] / 64, k0 = hl->e0[r] / 8;
                Act A[MR];
                for (int i = 0; i < nr; i++) {
                    const size_t k = (size_t)(j0 + i), eo = k * (INTER / 8), eg = k * 32;
                    A[i] = Act{P.he.data() + eo + k0, P.ho.data() + eo + k0, P.hsx.data() + eg + g0, P.hre.data() + eo + k0,
                               P.hro.data() + eo + k0, P.hrsx.data() + eg + g0, P.hs.data() + eg + g0, P.hrs.data() + eg + g0};
                }
                tile_rows<8>(nr, (const int32_t*)(b + S.dq) + (size_t)tt * (S.ni / 8) * 32,
                             (const uint16_t*)(b + S.ds) + (size_t)tt * (S.ni / 64) * 32, 32, 0, S.ni, A, f0, f1);
            }
            for (int i = 0; i < nr; i++) {
                vf_storeu(P.yp + (size_t)(j0 + i) * H + c0, f0[i]);
                vf_storeu(P.yp + (size_t)(j0 + i) * H + c0 + 16, f1[i]);
            }
        }
    }
    if (!P.y) return;
    barrier(bar4);
    for (int x = id; x < P.nx; x += nth) {  // per row: its pairs in order
        float* yr = P.y + (size_t)x * H;
        for (int c = 0; c < H; c += 16) {
            VF a = vf_zero();
            for (int q = P.roff[x]; q < P.roff[x + 1]; q++) a = vf_add(a, vf_loadu(P.yp + (size_t)P.rp[q] * H + c));
            vf_storeu(yr + c, a);
        }
    }
}

}  // namespace glm
