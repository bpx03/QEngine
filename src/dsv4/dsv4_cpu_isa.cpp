// DSV4 CPU expert lane kernels (see dsv4_cpu.h). Kernels adapted from bench/cpu_moe/cpu_moe.cpp (gate (a')).
// [dsv4-rel3-isa] compiled twice: DSV4_ISA_AVX512=1 (the Zen 4 set, the kernels as before) and DSV4_ISA_AVX512=0
// (x86-64-v3: the same kernels on 256-bit vectors, gguf_lane_avx2.h / gguf_pfcpu_avx2.h / the #else branches here).
// Every header with inline (shared) functions is included BEFORE the target pragma, so their out-of-line copies are
// plain x86-64 code in both objects (the linker keeps one of them for every caller); what follows the pragma is static
// or has a per-build name (CpuLane::work_avx512 / _avx2, namespace cpu_avx512 / cpu_avx2).
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <immintrin.h>

#include "dsv4.h"
#include "dsv4_cpu.h"
#include "gguf_iq.h"

#ifndef DSV4_ISA_AVX512
#error "build with -DDSV4_ISA_AVX512=1 (AVX-512 kernels) or =0 (AVX2 kernels)"
#endif
#if !defined(__GNUC__) || defined(__clang__)
#error "GCC only: the AVX2 kernels copy GCC's _mm512_reduce_add_ps order (bit-exact builds; review rel3 L2)"
#endif
// review rel3 H1 / M1: ISA lists, no arch= / tune= (GCC 13 refuses to inline the F16C intrinsic and the plain helpers
// into functions of another arch / tune); tuning comes from -mtune on the command line (build.sh), the same for the
// whole unit. The lists = cpu_isa()'s checks (dsv4_cpu.cpp)
#if DSV4_ISA_AVX512
#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512ifma,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,avx512bf16,gfni,vaes,vpclmulqdq,avx2,avx,fma,f16c,bmi,bmi2,lzcnt,movbe,popcnt")
#define ISA_FN(n) n##_avx512
#define ISA_NS cpu_avx512
#else
#pragma GCC target("avx2,avx,fma,f16c,bmi,bmi2,lzcnt,movbe,popcnt")
#define ISA_FN(n) n##_avx2
#define ISA_NS cpu_avx2
#endif

#if DSV4_ISA_AVX512
#include "gguf_lane.h"   // R16 [dsv4-gguf-iq]
#include "gguf_pfcpu.h"   // [dsv4-pf2-cpu]
#else
#include "gguf_lane_avx2.h"
#include "gguf_pfcpu_avx2.h"
namespace dsv4 { namespace pfc = pfc2; }
#endif

namespace dsv4 {

constexpr int HS = INTER / 32 + INTER / 128;   // per job: activation sums (<= per 32) then scales per 128

#if DSV4_ISA_AVX512
// lane helpers of this build: the prefetch touch (one load per 64 B line), the bf16 -> fp32 widening, the codebook lut
using PfAcc = __m512i;
static inline PfAcc pf_zero() { return _mm512_setzero_si512(); }
static inline PfAcc pf_or(PfAcc a, const uint8_t* p) { return _mm512_or_si512(a, _mm512_load_si512(p)); }
static inline long pf_sink(PfAcc a) { return _mm512_reduce_or_epi64(a); }
static inline void bf16_to_f32(const uint16_t* xb, int K, float* tmp) {
    for (int k = 0; k < K; k += 16)
        _mm512_storeu_ps(tmp + k, _mm512_castsi512_ps(_mm512_slli_epi32(
                                       _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(xb + k))), 16)));
}
using LutV = __m512i;
static inline LutV lut_load(const uint8_t* l) { return _mm512_loadu_si512(l); }

static inline __mmask16 tailmask(int n) { return n >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << n) - 1); }
static inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// FP4 e2m1 g32 / e8m0 (checkpoint bytes): nibble -> u8 (2w + 12), activations permuted per 128 [64 even][64 odd]
template <int M, int PFD = 0>
static void k_fp4v(const uint8_t* q, const uint8_t* s, int K, int r0, int r1, const int8_t* a, const float* asum,
                   const float* ascl, float* y, int ldy, int rlim = 0) {
    const int NB = K >> 7, G = K >> 5;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    __m512i pidx[4];
    for (int j = 0; j < 4; j++)
        pidx[j] = _mm512_add_epi32(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
                                   _mm512_set1_epi32(4 * j));
    const __m512 twelve = _mm512_set1_ps(12.f);
    alignas(64) float sf[192];
    alignas(64) float aex[MAXR][192];   // activation scale per 128, expanded per 32-group
    for (int i = 0; i < M; i++)
        for (int g = 0; g < G; g++) aex[i][g] = ascl[(size_t)i * NB + g / 4];
    for (int r = r0; r < r1; r++) {
        const uint8_t* qr = q + (size_t)r * (K >> 1);
        const uint8_t* sr = s + (size_t)r * G;
        // R13: software prefetch PFD rows ahead (one line per 64 B block): the host copies sit on 4 KB pages
        // (cudaHostAlloc), where the hardware streamers stop at every page; loads only, results unchanged
        const uint8_t* qp = PFD && r + PFD < rlim ? qr + (size_t)PFD * (K >> 1) : nullptr;
        if (qp) {   // review r13 L8: the whole scale row (G = K / 32 bytes: 2-3 lines; the last byte covers a straddle)
            for (int o = 0; o < G; o += 64) _mm_prefetch((const char*)(sr + (size_t)PFD * G + o), _MM_HINT_T0);
            _mm_prefetch((const char*)(sr + (size_t)PFD * G + G - 1), _MM_HINT_T0);
        }
        for (int g = 0; g < G; g += 16) {
            __m512i e = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tailmask(G - g), sr + g));
            _mm512_store_ps(sf + g, _mm512_castsi512_ps(_mm512_slli_epi32(e, 23)));
        }
        __m512 acc[M];
        for (int i = 0; i < M; i++) acc[i] = _mm512_setzero_ps();
        __m512 sv = _mm512_setzero_ps();
        for (int b = 0; b < NB; b++) {
            if ((b & 3) == 0) sv = _mm512_load_ps(sf + 4 * b);
            if (qp) _mm_prefetch((const char*)(qp + b * 64), _MM_HINT_T0);
            __m512i v = _mm512_loadu_si512(qr + b * 64);
            __m512i lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4));
            __m512i hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
            __m512 sc = _mm512_permutexvar_ps(pidx[b & 3], sv);
            for (int i = 0; i < M; i++) {
                const int8_t* ag = a + (size_t)i * K + b * 128;
                __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_loadu_si512(ag));
                d = _mm512_dpbusd_epi32(d, hi, _mm512_loadu_si512(ag + 64));
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_mul_ps(sc, _mm512_set1_ps(ascl[(size_t)i * NB + b])), acc[i]);
            }
        }
        for (int i = 0; i < M; i++) {
            __m512 t = acc[i];
            for (int g = 0; g < G; g += 16)
                t = _mm512_fnmadd_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(sf + g), twelve),
                                                   _mm512_maskz_loadu_ps(tailmask(G - g), aex[i] + g)),
                                     _mm512_maskz_loadu_ps(tailmask(G - g), asum + i * G + g), t);
            y[(size_t)i * ldy + r] = 0.5f * hsum(t);
        }
    }
}

// codebook g128 (3 bit: 48 B per group, HB; 2 bit: 32 B): codes -> u8 through vpermb (lut bytes 0..7, offset 128),
// fp16 scale per 128; result x post (codebook unit)
template <int M, bool HB>
static void k_cbv(const uint8_t* q, const uint16_t* s, int K, int r0, int r1, const int8_t* a, const float* asum,
                  const float* ascl, float* y, int ldy, __m512i lut, float post) {
    const int G = K >> 7;
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i four = _mm512_set1_epi8(4);
    alignas(64) uint16_t shl[32], shh[32];
    for (int i = 0; i < 32; i++) {
        shl[i] = i < 16 ? 0 : 2;
        shh[i] = i < 16 ? 4 : 6;
    }
    const __m512i vshl = _mm512_load_si512(shl), vshh = _mm512_load_si512(shh);
    const __m512 off = _mm512_set1_ps(128.f);
    constexpr int GB = HB ? 48 : 32;
    alignas(64) float sf[64];
    for (int r = r0; r < r1; r++) {
        const uint8_t* qr = q + (size_t)r * G * GB;
        const uint16_t* sr = s + (size_t)r * G;
        for (int g = 0; g < G; g += 16)
            _mm512_store_ps(sf + g, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailmask(G - g), sr + g)));
        __m512 acc[M];
        for (int i = 0; i < M; i++) acc[i] = _mm512_setzero_ps();
        for (int g = 0; g < G; g++) {
            const uint8_t* p = qr + g * GB;
            __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)p));
            __m512i lo = _mm512_and_si512(_mm512_srlv_epi16(v, vshl), m3);
            __m512i hi = _mm512_and_si512(_mm512_srlv_epi16(v, vshh), m3);
            if (HB) {
                uint64_t b0, b1;
                memcpy(&b0, p + 32, 8);
                memcpy(&b1, p + 40, 8);
                lo = _mm512_mask_add_epi8(lo, (__mmask64)b0, lo, four);
                hi = _mm512_mask_add_epi8(hi, (__mmask64)b1, hi, four);
            }
            lo = _mm512_permutexvar_epi8(lo, lut);
            hi = _mm512_permutexvar_epi8(hi, lut);
            for (int i = 0; i < M; i++) {
                const int8_t* ag = a + (size_t)i * K + g * 128;
                __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_loadu_si512(ag));
                d = _mm512_dpbusd_epi32(d, hi, _mm512_loadu_si512(ag + 64));
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_set1_ps(sf[g] * ascl[(size_t)i * G + g]), acc[i]);
            }
        }
        for (int i = 0; i < M; i++) {
            __m512 t = acc[i];
            for (int g = 0; g < G; g += 16)
                t = _mm512_fnmadd_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(sf + g), off),
                                                   _mm512_maskz_loadu_ps(tailmask(G - g), ascl + (size_t)i * G + g)),
                                     _mm512_maskz_loadu_ps(tailmask(G - g), asum + i * G + g), t);
            y[(size_t)i * ldy + r] = post * hsum(t);
        }
    }
}

// ---------------------------------------------------------------- activation prep

static float absmax(const float* v, int n) {
    __m512 m = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(v + i)));
    return _mm512_reduce_max_ps(m);
}
// int8 with a scale per 128 (scl [K / 128]); fp4 layout (permuted per 128, sums per 32) and plain layout (sums per 128)
static void quant_row(const float* v, int K, int8_t* q4, float* s32, int8_t* qp, float* s128, float* scl) {
    alignas(64) int8_t t8[8192];
    for (int g = 0; g < K / 128; g++) {
        float am = absmax(v + g * 128, 128);
        float s = am > 0 ? am / 127.f : 1.f;
        scl[g] = s;
        const __m512 inv = _mm512_set1_ps(1.f / s);
        for (int k = g * 128; k < g * 128 + 128; k += 16)
            _mm_storeu_si128((__m128i*)(t8 + k),
                             _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(v + k), inv))));
    }

#else   // ---------------------------------------------------------------- AVX2 build [dsv4-rel3-isa]
// the kernels above on two ymm halves (lanes 0-7 / 8-15 of each AVX-512 vector): same integer lane sums, same float
// operations per lane, the reduction in _mm512_reduce_add_ps's order (gl::reduce16, gguf_lane_avx2.h)

// lane helpers of this build (as the AVX-512 ones)
using PfAcc = __m256i;
static inline PfAcc pf_zero() { return _mm256_setzero_si256(); }
static inline PfAcc pf_or(PfAcc a, const uint8_t* p) { return _mm256_or_si256(a, _mm256_load_si256((const __m256i*)p)); }
static inline long pf_sink(PfAcc a) {
    const __m128i x = _mm_or_si128(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    return _mm_cvtsi128_si64(_mm_or_si128(x, _mm_unpackhi_epi64(x, x)));
}
static inline void bf16_to_f32(const uint16_t* xb, int K, float* tmp) {
    for (int k = 0; k < K; k += 8)
        _mm256_storeu_ps(tmp + k, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(xb + k))), 16)));
}
using LutV = __m256i;   // codebook bytes 0..7 (indices < 8) in both 128-bit lanes
static inline LutV lut_load(const uint8_t* l) { return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)l)); }

static inline float hsum(__m256 lo, __m256 hi) { return gl::reduce16(lo, hi); }
static inline __m256 loadn(const float* p, int n) {   // lanes >= n zero (as a maskz load)
    if (n >= 8) return _mm256_loadu_ps(p);
    if (n <= 0) return _mm256_setzero_ps();
    const __m256i m = _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    return _mm256_maskload_ps(p, m);
}
// u8 x s8 dot per 4 bytes for full-range u8 (the codebook bytes reach 255: vpmaddubsw would saturate): widened
static inline __m256i dot4w(__m256i g, __m256i a) {
    const __m256i x = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(g)), _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)));
    const __m256i y = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(g, 1)), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)));
    return _mm256_permute4x64_epi64(_mm256_hadd_epi32(x, y), 0xD8);
}

template <int M, int PFD = 0>
static void k_fp4v(const uint8_t* q, const uint8_t* s, int K, int r0, int r1, const int8_t* a, const float* asum,
                   const float* ascl, float* y, int ldy, int rlim = 0) {
    const int NB = K >> 7, G = K >> 5;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m256 twelve = _mm256_set1_ps(12.f);
    alignas(64) float sf[192];
    alignas(64) float aex[MAXR][192];
    for (int i = 0; i < M; i++)
        for (int g = 0; g < G; g++) aex[i][g] = ascl[(size_t)i * NB + g / 4];
    const int GP = (G + 15) & ~15;
    for (int r = r0; r < r1; r++) {
        const uint8_t* qr = q + (size_t)r * (K >> 1);
        const uint8_t* sr = s + (size_t)r * G;
        const uint8_t* qp = PFD && r + PFD < rlim ? qr + (size_t)PFD * (K >> 1) : nullptr;
        if (qp) {
            for (int o = 0; o < G; o += 64) _mm_prefetch((const char*)(sr + (size_t)PFD * G + o), _MM_HINT_T0);
            _mm_prefetch((const char*)(sr + (size_t)PFD * G + G - 1), _MM_HINT_T0);
        }
        for (int g = 0; g < GP; g++) {
            const uint32_t e = g < G ? (uint32_t)sr[g] << 23 : 0u;
            memcpy(sf + g, &e, 4);
        }
        __m256 acl[M], ach[M];
        for (int i = 0; i < M; i++) acl[i] = ach[i] = _mm256_setzero_ps();
        for (int b = 0; b < NB; b++) {
            if (qp) _mm_prefetch((const char*)(qp + b * 64), _MM_HINT_T0);
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(qr + b * 64)), v1 = _mm256_loadu_si256((const __m256i*)(qr + b * 64 + 32));
            const __m256i lo0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v0, m4)), lo1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v1, m4));
            const __m256i hi0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4));
            const __m256i hi1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4));
            const float s0 = sf[4 * b], s1 = sf[4 * b + 1], s2 = sf[4 * b + 2], s3 = sf[4 * b + 3];
            const __m256 scl = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1), sch = _mm256_setr_ps(s2, s2, s2, s2, s3, s3, s3, s3);
            for (int i = 0; i < M; i++) {
                const int8_t* ag = a + (size_t)i * K + b * 128;
                const __m256i dl = _mm256_add_epi32(gl::dot4(lo0, _mm256_loadu_si256((const __m256i*)ag)),
                                                    gl::dot4(hi0, _mm256_loadu_si256((const __m256i*)(ag + 64))));
                const __m256i dh = _mm256_add_epi32(gl::dot4(lo1, _mm256_loadu_si256((const __m256i*)(ag + 32))),
                                                    gl::dot4(hi1, _mm256_loadu_si256((const __m256i*)(ag + 96))));
                const __m256 as = _mm256_set1_ps(ascl[(size_t)i * NB + b]);
                acl[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), _mm256_mul_ps(scl, as), acl[i]);
                ach[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), _mm256_mul_ps(sch, as), ach[i]);
            }
        }
        for (int i = 0; i < M; i++) {
            __m256 tl = acl[i], th = ach[i];
            for (int g = 0; g < G; g += 16) {
                tl = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(sf + g), twelve), loadn(aex[i] + g, G - g)),
                                      loadn(asum + i * G + g, G - g), tl);
                th = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(sf + g + 8), twelve), loadn(aex[i] + g + 8, G - g - 8)),
                                      loadn(asum + i * G + g + 8, G - g - 8), th);
            }
            y[(size_t)i * ldy + r] = 0.5f * hsum(tl, th);
        }
    }
}

template <int M, bool HB>
static void k_cbv(const uint8_t* q, const uint16_t* s, int K, int r0, int r1, const int8_t* a, const float* asum,
                  const float* ascl, float* y, int ldy, __m256i lut, float post) {
    const int G = K >> 7;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i four = _mm256_set1_epi8(4);
    const __m256 off = _mm256_set1_ps(128.f);
    constexpr int GB = HB ? 48 : 32;
    alignas(64) float sf[64];
    const int GP = (G + 15) & ~15;
    for (int r = r0; r < r1; r++) {
        const uint8_t* qr = q + (size_t)r * G * GB;
        const uint16_t* sr = s + (size_t)r * G;
        for (int g = 0; g < GP; g++) sf[g] = g < G ? _cvtsh_ss(sr[g]) : 0.f;
        __m256 acl[M], ach[M];
        for (int i = 0; i < M; i++) acl[i] = ach[i] = _mm256_setzero_ps();
        for (int g = 0; g < G; g++) {
            const uint8_t* p = qr + g * GB;
            const __m256i v = _mm256_loadu_si256((const __m256i*)p);
            __m256i lo0 = _mm256_and_si256(v, m3), lo1 = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
            __m256i hi0 = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3), hi1 = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
            if (HB) {
                uint32_t b[4];
                memcpy(b, p + 32, 16);
                lo0 = _mm256_add_epi8(lo0, _mm256_and_si256(gl::smask32(b[0]), four));
                lo1 = _mm256_add_epi8(lo1, _mm256_and_si256(gl::smask32(b[1]), four));
                hi0 = _mm256_add_epi8(hi0, _mm256_and_si256(gl::smask32(b[2]), four));
                hi1 = _mm256_add_epi8(hi1, _mm256_and_si256(gl::smask32(b[3]), four));
            }
            lo0 = _mm256_shuffle_epi8(lut, lo0);
            lo1 = _mm256_shuffle_epi8(lut, lo1);
            hi0 = _mm256_shuffle_epi8(lut, hi0);
            hi1 = _mm256_shuffle_epi8(lut, hi1);
            for (int i = 0; i < M; i++) {
                const int8_t* ag = a + (size_t)i * K + g * 128;
                const __m256i dl = _mm256_add_epi32(dot4w(lo0, _mm256_loadu_si256((const __m256i*)ag)),
                                                    dot4w(hi0, _mm256_loadu_si256((const __m256i*)(ag + 64))));
                const __m256i dh = _mm256_add_epi32(dot4w(lo1, _mm256_loadu_si256((const __m256i*)(ag + 32))),
                                                    dot4w(hi1, _mm256_loadu_si256((const __m256i*)(ag + 96))));
                const __m256 w = _mm256_set1_ps(sf[g] * ascl[(size_t)i * G + g]);
                acl[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), w, acl[i]);
                ach[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), w, ach[i]);
            }
        }
        for (int i = 0; i < M; i++) {
            __m256 tl = acl[i], th = ach[i];
            for (int g = 0; g < G; g += 16) {
                tl = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(sf + g), off), loadn(ascl + (size_t)i * G + g, G - g)),
                                      loadn(asum + i * G + g, G - g), tl);
                th = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(sf + g + 8), off), loadn(ascl + (size_t)i * G + g + 8, G - g - 8)),
                                      loadn(asum + i * G + g + 8, G - g - 8), th);
            }
            y[(size_t)i * ldy + r] = post * hsum(tl, th);
        }
    }
}

// ---------------------------------------------------------------- activation prep

static float absmax(const float* v, int n) {   // (max is exact: any order; lanes as the AVX-512 build)
    const __m256 ab = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        lo = _mm256_max_ps(lo, _mm256_and_ps(_mm256_loadu_ps(v + i), ab));
        hi = _mm256_max_ps(hi, _mm256_and_ps(_mm256_loadu_ps(v + i + 8), ab));
    }
    const __m256 t3 = _mm256_max_ps(hi, lo);
    const __m128 t6 = _mm_max_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
    const __m128 t8 = _mm_max_ps(t6, _mm_shuffle_ps(t6, t6, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128 t10 = _mm_max_ps(t8, _mm_shuffle_ps(t8, t8, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtss_f32(t10);
}
static void quant_row(const float* v, int K, int8_t* q4, float* s32, int8_t* qp, float* s128, float* scl) {
    alignas(64) int8_t t8[8192];
    for (int g = 0; g < K / 128; g++) {
        float am = absmax(v + g * 128, 128);
        float s = am > 0 ? am / 127.f : 1.f;
        scl[g] = s;
        const __m256 inv = _mm256_set1_ps(1.f / s);
        for (int k = g * 128; k < g * 128 + 128; k += 16) {   // round to nearest, saturate to int8 (= vpmovsdb)
            const __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + k), inv));
            const __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + k + 8), inv));
            const __m256i w = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
            _mm_storeu_si128((__m128i*)(t8 + k), _mm_packs_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1)));
        }
    }
#endif
    if (qp) memcpy(qp, t8, K);
    if (s32)
        for (int g = 0; g < K / 32; g++) {
            int su = 0;
            for (int j = 0; j < 32; j++) su += t8[g * 32 + j];
            s32[g] = (float)su;
        }
    if (s128)
        for (int g = 0; g < K / 128; g++) {
            int su = 0;
            for (int j = 0; j < 128; j++) su += t8[g * 128 + j];
            s128[g] = (float)su;
        }
    if (q4)
        for (int b = 0; b < K / 128; b++)
            for (int j = 0; j < 64; j++) {
                q4[b * 128 + j] = t8[b * 128 + 2 * j];
                q4[b * 128 + 64 + j] = t8[b * 128 + 2 * j + 1];
            }
}
// e4m3 round to nearest even of |v| <= 448 (torch .to(float8_e4m3fn))
static inline float e4m3_rne(float v) {
    float a = fabsf(v);
    if (a == 0.f) return v;
    int e;
    frexpf(a, &e);
    int ex = std::max(e - 1, -6);
    float q = ldexpf(1.f, ex - 3);
    return copysignf(nearbyintf(a / q) * q, v);
}
static inline float pow2ceil(float v) {
    int e;
    float m = frexpf(v, &e);
    if (m == 0.5f) e -= 1;
    return ldexpf(1.f, e);
}
static void q_fp8_row(float* v, int K) {
    for (int b = 0; b < K; b += 32) {
        float am = 1e-4f;
        for (int j = 0; j < 32; j++) am = std::max(am, fabsf(v[b + j]));
        float s = pow2ceil(am * (1.f / 448.f));
        for (int j = 0; j < 32; j++) v[b + j] = e4m3_rne(std::min(std::max(v[b + j] / s, -448.f), 448.f)) * s;
    }
}

// rows [b0, b1) in 16-row blocks of one matrix for thread id: the same split in run() and prefetch()
static inline void split_blocks(int nblk, int id, int nt, int& b0, int& b1) {
    b0 = (int)((long long)nblk * id / nt);
    b1 = (int)((long long)nblk * (id + 1) / nt);
}
static inline size_t row_bytes(int fmt, int K) {
    if (is_iq(fmt)) return (size_t)(K / 256) * gg_cbytes(fmt);   // R16 [dsv4-gguf-iq]
    return fmt == XF_FP4 ? (size_t)K / 2 : (size_t)(K / 128) * (fmt == XF_CB3 ? 48 : 32);
}
static inline size_t srow_bytes(int fmt, int K) {
    if (is_iq(fmt)) return (size_t)(K / 256) * 2;
    return fmt == XF_FP4 ? (size_t)K / 32 : (size_t)(K / 128) * 2;
}

void CpuLane::ISA_FN(prefetch_work)(int id, uint64_t g) {
    // seqlock-style copy of the parameters (review r4 M1): a stale worker never reads a slot being rewritten
    Pf P;
    P.n = pf_[g & 1].n;
    P.stop = pf_[g & 1].stop;
    P.stopval = pf_[g & 1].stopval;
    for (int i = 0; i < P.n && i < 2 * MAXR * TOPK; i++) P.xs[i] = pf_[g & 1].xs[i];
    std::atomic_thread_fence(std::memory_order_acquire);
    if (gen_.load(std::memory_order_acquire) != g) return;
    const int nt = nt_;
    PfAcc acc = pf_zero();
    long lines = 0;
    const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
    for (int x = 0; x < P.n; x++) {
        if (*P.stop == P.stopval) break;
        MatView mv[3];
        const int fm = P.xs[x].fmt;   // read once
        expert_views((XFmt)fm, P.xs[x].base, mv);
        for (int m = 0; m < 3; m++) {
            int b0, b1;
            split_blocks(Nm[m] / 16, id, nt, b0, b1);
            const int mf = mat_fmt((XFmt)fm, m);   // R16: per-matrix format (M23)
            if (P.xs[x].nat[0]) {   // [dsv4-rel-ram]: GGUF rows in place (blocks of cbytes + 2, d inside)
                const size_t rb = (size_t)(Km[m] / 256) * (gg_cbytes(mf) + 2);
                const uint8_t* q0 = P.xs[x].nat[m] + (size_t)b0 * 16 * rb;
                const uint8_t* q1 = P.xs[x].nat[m] + (size_t)b1 * 16 * rb;
                int chk = 0;
                for (const uint8_t* p = (const uint8_t*)((uintptr_t)q0 & ~(uintptr_t)63); p < q1; p += 64) {
                    if (++chk == 128) {
                        chk = 0;
                        if (*P.stop == P.stopval || gen_.load(std::memory_order_relaxed) != g) goto done;
                    }
                    acc = pf_or(acc, p);
                    lines++;
                }
                continue;
            }
            const size_t rb = row_bytes(mf, Km[m]), sb = srow_bytes(mf, Km[m]);
            const uint8_t* q0 = mv[m].q + (size_t)b0 * 16 * rb;
            const uint8_t* q1 = mv[m].q + (size_t)b1 * 16 * rb;
            int chk = 0;
            for (const uint8_t* p = (const uint8_t*)((uintptr_t)q0 & ~(uintptr_t)63); p < q1; p += 256) {
                if (++chk == 32) {   // every 8 KB (review r4 M2): stop as soon as the routing has landed
                    chk = 0;
                    if (*P.stop == P.stopval || gen_.load(std::memory_order_relaxed) != g) goto done;
                }
                acc = pf_or(acc, p);
                if (p + 64 < q1) acc = pf_or(acc, p + 64);
                if (p + 128 < q1) acc = pf_or(acc, p + 128);
                if (p + 192 < q1) acc = pf_or(acc, p + 192);
                lines += 4;
            }
            const uint8_t* s0 = (const uint8_t*)mv[m].s + (size_t)b0 * 16 * sb;
            const uint8_t* s1 = (const uint8_t*)mv[m].s + (size_t)b1 * 16 * sb;
            for (const uint8_t* p = (const uint8_t*)((uintptr_t)s0 & ~(uintptr_t)63); p < s1; p += 64) {
                acc = pf_or(acc, p);
                lines++;
            }
        }
    }
done:
    alignas(64) static thread_local volatile long sink;
    sink = pf_sink(acc);
    if (id == 0) pf_lines = lines;
}

// [dsv4-dec1-lmr] multi-row lane groups (default on; DSV4_NOLMR=1 = one row at a time, the old path)
static bool lmr_on() {
    static const bool on = !(getenv("DSV4_NOLMR") && atoi(getenv("DSV4_NOLMR")) != 0);
    return on;
}
// R16 [dsv4-gguf-iq] review M6: the plane-order copy of an activation row (gate / up: slot = row; down: MAXR + job),
// made once per run (generation g) and thread, not per 16-row block; nullptr when the format has no plane path
static const int8_t* gg_ap(const void* lane, uint64_t g, int slot, int kind, const int8_t* src, int K) {
    if (!kind) return nullptr;
    constexpr int NS = MAXR + MAXR * TOPK;
    static thread_local std::vector<int8_t> buf;
    static thread_local uint8_t done[NS];
    static thread_local uint64_t gen = ~0ull;
    static thread_local const void* own = nullptr;   // review r24 L5: keyed by lane and generation (a new lane restarts at 1)
    if (buf.empty()) buf.resize((size_t)NS * GG_PERM_MAXK + 64);
    if (gen != g || own != lane) {
        gen = g;
        own = lane;
        memset(done, 0, sizeof done);
    }
    int8_t* dst = (int8_t*)(((uintptr_t)buf.data() + 63) & ~(uintptr_t)63) + (size_t)slot * GG_PERM_MAXK;
    if (done[slot] != kind) {
        gg_perm_act(kind, src, K, dst);
        done[slot] = (uint8_t)kind;
    }
    return dst;
}

#define DISPATCH_M(MM, CALL)                                    \
    switch (MM) {                                               \
        case 1: { constexpr int MT = 1; CALL; } break;          \
        case 2: { constexpr int MT = 2; CALL; } break;          \
        case 3: { constexpr int MT = 3; CALL; } break;          \
        case 4: { constexpr int MT = 4; CALL; } break;          \
        case 5: { constexpr int MT = 5; CALL; } break;          \
        case 6: { constexpr int MT = 6; CALL; } break;          \
        case 7: { constexpr int MT = 7; CALL; } break;          \
        case 8: { constexpr int MT = 8; CALL; } break;          \
    }

// R24 [dsv4-r24-steal]: a thread takes the units of its own range from the front, then units of the other threads'
// ranges from their back (its own CCD first, 6 cores per CCD on this box), until every range is empty. A unit writes
// only its own 16 output rows of one job, with the same kernel call as the static split: results are bit-identical
template <class F>
static void steal_loop(CpuLane::Steal* st, int id, int nt, const F& f, std::atomic<long>* stolen) {
    long ns = 0;   // review r24 L3: counted locally, added once (only with DSV4_LANEPROF)
    for (;;) {   // own range, front
        uint64_t v = st[id].fb.load(std::memory_order_relaxed);
        const uint32_t fr = (uint32_t)(v >> 32), bk = (uint32_t)v;
        if (fr >= bk) break;
        if (st[id].fb.compare_exchange_weak(v, v + (1ull << 32), std::memory_order_relaxed)) f(id, (int)fr);
    }
    const int cc = 6, c0 = id / cc * cc;
    for (int k = 1; k < nt; k++) {
        int vic = k < cc ? c0 + (id - c0 + k) % cc : (c0 + cc + (k - cc)) % nt;
        if (k < cc && vic >= nt) continue;
        if (k >= cc && vic >= c0 && vic < c0 + cc) continue;
        for (;;) {   // victim's range, back
            uint64_t v = st[vic].fb.load(std::memory_order_relaxed);
            const uint32_t fr = (uint32_t)(v >> 32), bk = (uint32_t)v;
            if (fr >= bk) break;
            if (st[vic].fb.compare_exchange_weak(v, v - 1, std::memory_order_relaxed)) {
                f(vic, (int)bk - 1);
                ns++;
            }
        }
    }
    if (stolen && ns) stolen->fetch_add(ns, std::memory_order_relaxed);
}

void CpuLane::ISA_FN(work)(int id, uint64_t g) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int nt = nt_;
    uint64_t nb = barbase_[g & 15];   // barriers passed (5 per run; the counter only grows)
    static const bool prof = getenv("DSV4_LANEPROF") && atoi(getenv("DSV4_LANEPROF")) != 0;   // R24 [dsv4-r24-laneprof]
    auto tp = t0;
    int ph = 0;
    auto bar = [&] {
        if (prof) {
            const auto tn = clk::now();
            prof_[id].ph[ph++] += std::chrono::duration<double, std::milli>(tn - tp).count();
            tp = tn;
        }
        nb++;
        uint64_t target = nb * (uint64_t)nt;
        bar_.fetch_add(1, std::memory_order_acq_rel);
        while (bar_.load(std::memory_order_acquire) < target) _mm_pause();
        if (prof) {
            const auto ta = clk::now();
            prof_[id].wait += std::chrono::duration<double, std::milli>(ta - tp).count();
            tp = ta;
            if (ph == 5) prof_[id].n++;
        }
    };
    // phase 0: activation prep. R13: (row, 128-column group) units over the threads; quant_row works per group of
    // 128 (scale, sums per 32 / 128, the FP4 permutation), so a group slice gives the same bytes as the whole row
    static const bool par_off = getenv("DSV4_NOLANEPAR") != nullptr;
    static const bool swpf = getenv("DSV4_NOSWPF") == nullptr;   // R13: software prefetch in the FP4 kernel
    // R20 [dsv4-r20-xh16]: a bf16 xq (exact: FP8-rounded values) widened to fp32 per 128-column group first
    auto xrow = [&](size_t off, int K, float* tmp) -> const float* {
        if (!x16_) return xq_ + off;
        const uint16_t* xb = (const uint16_t*)xq_ + off;
        bf16_to_f32(xb, K, tmp);
        return tmp;
    };
    alignas(64) float xtmp[D];
    if (par_off) {
    for (int r = id; r < R_; r += nt)
        quant_row(xrow((size_t)r * D, D, xtmp), D, &aq4_[(size_t)r * D], &as32_[(size_t)r * D / 32], &aq_[(size_t)r * D],
                  &as128_[(size_t)r * D / 128], &ascl_[(size_t)r * D / 128]);
    } else {
        constexpr int NG0 = D / 128;
        for (int u = id; u < R_ * NG0; u += nt) {
            const int r = u / NG0;
            const size_t c0 = (size_t)(u % NG0) * 128;
            quant_row(xrow((size_t)r * D + c0, 128, xtmp), 128, &aq4_[(size_t)r * D + c0], &as32_[((size_t)r * D + c0) / 32],
                      &aq_[(size_t)r * D + c0], &as128_[((size_t)r * D + c0) / 128], &ascl_[((size_t)r * D + c0) / 128]);
        }
    }
    // R24 [dsv4-r24-steal]: this thread's unit ranges of phases 1 and 3 (front | back), set before the barrier, so a
    // thief never sees a previous run's range; units = (group, matrix, 16-row block) of its static split
    const int ng = (int)gs_.size() - 1;
    if (steal_) {
        int a0, a1, d0, d1;
        split_blocks(INTER / 16, id, nt, a0, a1);
        split_blocks(D / 16, id, nt, d0, d1);
        st1_[id].fb.store((uint64_t)(ng * 2 * (a1 - a0)), std::memory_order_relaxed);
        st3_[id].fb.store((uint64_t)(ng * (d1 - d0)), std::memory_order_relaxed);
    }
    bar();
    auto t1 = clk::now();
    const LutV lut3 = lut_load(lut3_), lut2 = lut_load(lut2_);
    // phase 1: gate / up, 16-row blocks of (job, matrix 0/1)
    const int BG = INTER / 16;
    int gb0, gb1;
    split_blocks(BG, id, nt, gb0, gb1);   // per expert matrix: this thread's rows (as prefetch_work)
    auto unit1 = [&](int gm, int b, int gb1) {
    if (lmr_on() && gs_[gm / 2 + 1] - gs_[gm / 2] >= 2) {   // [dsv4-dec1-lmr] the group's rows in one pass over its weights
        const int m = gm % 2, g0 = gs_[gm / 2], gn = gs_[gm / 2 + 1] - g0;
        const CpuJob& J0 = jobs_[gj_[g0]];
        const XFmt mf = mat_fmt((XFmt)J0.fmt, m);
        const int kind = gg_plane_kind(mf, D);
        bool all = is_iq(mf) && kind;
        for (int c0 = 0; all && c0 < gn; c0 += 6) {
            const int n = std::min(6, gn - c0);
            const int8_t* ap[6];
            const float* as[6];
            float* yo[6];
            for (int i = 0; i < n; i++) {
                const int j = gj_[g0 + c0 + i], r = jobs_[j].row;
                ap[i] = gg_ap(this, g, r, kind, &aq_[(size_t)r * D], D);
                as[i] = &ascl_[(size_t)r * D / 128];
                yo[i] = &gu_[(size_t)j * 2 * INTER + m * INTER];
            }
            MatView mv[3];
            expert_views((XFmt)J0.fmt, J0.base, mv);
            const bool nat = J0.nat[0] != nullptr;
            if (n == 1 || !k_ggv_rows(mf, nat, nat ? J0.nat[m] : mv[m].q, nat ? nullptr : (const uint16_t*)mv[m].s, D,
                                      b * 16, b * 16 + 16, n, ap, as, yo)) {
                if (c0 == 0) { all = false; break; }   // nothing written yet: the one-row path below
                for (int i = 0; i < n; i++) {   // (a single trailing job of a chunked group)
                    if (nat) k_ggv_nat(mf, J0.nat[m], D, b * 16, b * 16 + 16, nullptr, as[i], yo[i], ap[i]);
                    else k_ggv_any(mf, mv[m].q, (const uint16_t*)mv[m].s, D, b * 16, b * 16 + 16, nullptr, as[i], yo[i], ap[i]);
                }
            }
        }
        if (all) return;
    }
    for (int gi = gs_[gm / 2]; gi < gs_[gm / 2 + 1]; gi++) {
        int j = gj_[gi], m = gm % 2;
        const CpuJob& J = jobs_[j];
        MatView mv[3];
        expert_views((XFmt)J.fmt, J.base, mv);
        float* yo = &gu_[(size_t)j * 2 * INTER + m * INTER];
        int r = J.row;
        const XFmt mf = mat_fmt((XFmt)J.fmt, m);   // R16: M23 = cb2 gate / up
        if (mf == XF_FP4) {
            if (swpf)
            k_fp4v<1, 2>(mv[m].q, (const uint8_t*)mv[m].s, D, b * 16, b * 16 + 16, &aq4_[(size_t)r * D],
                      &as32_[(size_t)r * D / 32], &ascl_[(size_t)r * D / 128], yo, 0, gb1 * 16);
            else
            k_fp4v<1>(mv[m].q, (const uint8_t*)mv[m].s, D, b * 16, b * 16 + 16, &aq4_[(size_t)r * D],
                      &as32_[(size_t)r * D / 32], &ascl_[(size_t)r * D / 128], yo, 0);
        }
        else if (J.nat[0])   // [dsv4-rel-ram]: no host copy, the GGUF bytes in place
            k_ggv_nat(mf, J.nat[m], D, b * 16, b * 16 + 16, &aq_[(size_t)r * D], &ascl_[(size_t)r * D / 128], yo,
                      gg_ap(this, g, r, gg_plane_kind(mf, D), &aq_[(size_t)r * D], D));
        else if (is_iq(mf))   // R16 [dsv4-gguf-iq]: plain int8 activations, scale per 128 (+ plane order, once per row)
            k_ggv_any(mf, mv[m].q, (const uint16_t*)mv[m].s, D, b * 16, b * 16 + 16, &aq_[(size_t)r * D],
                      &ascl_[(size_t)r * D / 128], yo, gg_ap(this, g, r, gg_plane_kind(mf, D), &aq_[(size_t)r * D], D));
        else if (mf == XF_CB3)
            k_cbv<1, true>(mv[m].q, (const uint16_t*)mv[m].s, D, b * 16, b * 16 + 16, &aq_[(size_t)r * D],
                           &as128_[(size_t)r * D / 128], &ascl_[(size_t)r * D / 128], yo, 0, lut3, eff3_);
        else
            k_cbv<1, false>(mv[m].q, (const uint16_t*)mv[m].s, D, b * 16, b * 16 + 16, &aq_[(size_t)r * D],
                            &as128_[(size_t)r * D / 128], &ascl_[(size_t)r * D / 128], yo, 0, lut2, eff2_);
    }
    };
    if (!steal_) {
        for (int gm = 0; gm < ng * 2; gm++)
            for (int b = gb0; b < gb1; b++) unit1(gm, b, gb1);
    } else
        steal_loop(st1_.get(), id, nt, [&](int v, int u) {   // unit u of thread v's range: (gm, block) in its order
            int a0, a1;
            split_blocks(BG, v, nt, a0, a1);
            unit1(u / (a1 - a0), a0 + u % (a1 - a0), a1);
        }, prof ? &stolen1 : nullptr);
    bar();
    auto t2 = clk::now();
    // phase 2: GLU, FP8 rounding (as the reference's lin_q on h), int8 per row in the job's layout.
    // R13: (job, 128-column group) units over the threads (was one thread per job, ~30 us of a ~160 us layer);
    // the same scalar arithmetic per element, q_fp8 per 32 and quant per 128 inside the group: identical bytes
    constexpr int NG2 = INTER / 128;
    if (!par_off)
    for (int u = id; u < nj_ * NG2; u += nt) {
        const int j = u / NG2, c0 = (u % NG2) * 128;
        const float* g = &gu_[(size_t)j * 2 * INTER];
        float* h = &hs_[(size_t)j * INTER];
        const float w = jobs_[j].w;
        for (int c = c0; c < c0 + 128; c++) {
            float gg = std::min(g[c], CLAMP);
            float uu = std::min(std::max(g[INTER + c], -CLAMP), CLAMP);
            h[c] = w * (gg / (1.f + std::exp(-gg)) * uu);
        }
        q_fp8_row(h + c0, 128);
        float* sums = &hscl_[(size_t)j * HS];
        if (mat_fmt((XFmt)jobs_[j].fmt, 2) == XF_FP4)
            quant_row(h + c0, 128, &hq_[(size_t)j * INTER + c0], sums + c0 / 32, nullptr, nullptr,
                      &sums[INTER / 32 + c0 / 128]);
        else
            quant_row(h + c0, 128, nullptr, nullptr, &hq_[(size_t)j * INTER + c0], sums + c0 / 128,
                      &sums[INTER / 32 + c0 / 128]);
    }
    if (par_off)
    for (int j = id; j < nj_; j += nt) {
        float* g = &gu_[(size_t)j * 2 * INTER];
        float* h = &hs_[(size_t)j * INTER];
        float w = jobs_[j].w;
        for (int c = 0; c < INTER; c++) {
            float gg = std::min(g[c], CLAMP);
            float uu = std::min(std::max(g[INTER + c], -CLAMP), CLAMP);
            h[c] = w * (gg / (1.f + std::exp(-gg)) * uu);
        }
        q_fp8_row(h, INTER);
        float* sums = &hscl_[(size_t)j * HS];
        if (mat_fmt((XFmt)jobs_[j].fmt, 2) == XF_FP4)
            quant_row(h, INTER, &hq_[(size_t)j * INTER], sums, nullptr, nullptr, &sums[INTER / 32]);
        else
            quant_row(h, INTER, nullptr, nullptr, &hq_[(size_t)j * INTER], sums, &sums[INTER / 32]);
    }
    bar();
    auto t3 = clk::now();
    // phase 3: down, 16-row blocks of (job)
    const int BD = D / 16;
    int db0, db1;
    split_blocks(BD, id, nt, db0, db1);
    auto unit3 = [&](int gg, int b, int db1) {
    if (lmr_on() && gs_[gg + 1] - gs_[gg] >= 2) {   // [dsv4-dec1-lmr] the group's jobs in one pass over the down weights
        const int g0 = gs_[gg], gn = gs_[gg + 1] - g0;
        const CpuJob& J0 = jobs_[gj_[g0]];
        const XFmt mf = mat_fmt((XFmt)J0.fmt, 2);
        const int kind = gg_plane_kind(mf, INTER);
        bool all = is_iq(mf) && kind;
        for (int c0 = 0; all && c0 < gn; c0 += 6) {
            const int n = std::min(6, gn - c0);
            const int8_t* ap[6];
            const float* as[6];
            float* yo[6];
            for (int i = 0; i < n; i++) {
                const int j = gj_[g0 + c0 + i];
                float* sums = &hscl_[(size_t)j * HS];
                ap[i] = gg_ap(this, g, MAXR + j, kind, &hq_[(size_t)j * INTER], INTER);
                as[i] = &sums[INTER / 32];
                yo[i] = &yj_[(size_t)j * D];
            }
            MatView mv[3];
            expert_views((XFmt)J0.fmt, J0.base, mv);
            const bool nat = J0.nat[0] != nullptr;
            if (n == 1 || !k_ggv_rows(mf, nat, nat ? J0.nat[2] : mv[2].q, nat ? nullptr : (const uint16_t*)mv[2].s, INTER,
                                      b * 16, b * 16 + 16, n, ap, as, yo)) {
                if (c0 == 0) { all = false; break; }
                for (int i = 0; i < n; i++) {
                    if (nat) k_ggv_nat(mf, J0.nat[2], INTER, b * 16, b * 16 + 16, nullptr, as[i], yo[i], ap[i]);
                    else k_ggv_any(mf, mv[2].q, (const uint16_t*)mv[2].s, INTER, b * 16, b * 16 + 16, nullptr, as[i], yo[i], ap[i]);
                }
            }
        }
        if (all) return;
    }
    for (int gi = gs_[gg]; gi < gs_[gg + 1]; gi++) {
        int j = gj_[gi];
        const CpuJob& J = jobs_[j];
        MatView mv[3];
        expert_views((XFmt)J.fmt, J.base, mv);
        float* sums = &hscl_[(size_t)j * HS];
        float* yo = &yj_[(size_t)j * D];
        const XFmt mf = mat_fmt((XFmt)J.fmt, 2);   // R16: M23 = cb3 down
        if (mf == XF_FP4) {
            if (swpf)
            k_fp4v<1, 4>(mv[2].q, (const uint8_t*)mv[2].s, INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER], sums,
                         &sums[INTER / 32], yo, 0, db1 * 16);
            else
            k_fp4v<1>(mv[2].q, (const uint8_t*)mv[2].s, INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER], sums,
                      &sums[INTER / 32], yo, 0);
        }
        else if (J.nat[0])   // [dsv4-rel-ram]
            k_ggv_nat(mf, J.nat[2], INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER], &sums[INTER / 32], yo,
                      gg_ap(this, g, MAXR + j, gg_plane_kind(mf, INTER), &hq_[(size_t)j * INTER], INTER));
        else if (is_iq(mf))   // R16 [dsv4-gguf-iq]
            k_ggv_any(mf, mv[2].q, (const uint16_t*)mv[2].s, INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER],
                      &sums[INTER / 32], yo, gg_ap(this, g, MAXR + j, gg_plane_kind(mf, INTER), &hq_[(size_t)j * INTER], INTER));
        else if (mf == XF_CB3)
            k_cbv<1, true>(mv[2].q, (const uint16_t*)mv[2].s, INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER],
                           sums, &sums[INTER / 32], yo, 0, lut3, eff3_);
        else
            k_cbv<1, false>(mv[2].q, (const uint16_t*)mv[2].s, INTER, b * 16, b * 16 + 16, &hq_[(size_t)j * INTER],
                            sums, &sums[INTER / 32], yo, 0, lut2, eff2_);
    }
    };
    if (!steal_) {
        for (int gg = 0; gg < ng; gg++)
            for (int b = db0; b < db1; b++) unit3(gg, b, db1);
    } else
        steal_loop(st3_.get(), id, nt, [&](int v, int u) {
            int a0, a1;
            split_blocks(BD, v, nt, a0, a1);
            unit3(u / (a1 - a0), a0 + u % (a1 - a0), a1);
        }, prof ? &stolen3 : nullptr);
    bar();
    auto t4 = clk::now();
    // phase 4: combine per row in job order, columns split over threads
    const int c0 = (int)((long long)D * id / nt) & ~15, c1 = id == nt - 1 ? D : ((int)((long long)D * (id + 1) / nt) & ~15);
    for (int r = 0; r < R_; r++) {
        float* yr = y_ + (size_t)r * D;
        for (int c = c0; c < c1; c++) yr[c] = 0.f;
        for (int j = 0; j < nj_; j++)
            if (jobs_[j].row == r) {
                const float* s = &yj_[(size_t)j * D];
                for (int c = c0; c < c1; c++) yr[c] += s[c];
            }
    }
    bar();
    if (id == 0) {
        auto t5 = clk::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        last_ms[0] = ms(t0, t1);
        last_ms[1] = ms(t1, t2);
        last_ms[2] = ms(t2, t3);
        last_ms[3] = ms(t3, t4);
        last_ms[4] = ms(t4, t5);
    }
}

// ---------------------------------------------------------------- [dsv4-pf2-cpu] prefill experts on the CPU
// [dsv4-pf8-xqh] E4M3 code -> float (the CUDA E4M3 decode: e = 0 subnormals m * 2^-9, 0x7f / 0xff NaN)
static const std::array<float, 256> g_e4m3 = [] {
    std::array<float, 256> t{};
    for (int c = 0; c < 256; c++) {
        const int ex = (c >> 3) & 15, mn = c & 7;
        float v = ex ? std::ldexp(1.f + mn / 8.f, ex - 7) : std::ldexp((float)mn, -9);
        if ((c & 127) == 127) v = std::numeric_limits<float>::quiet_NaN();
        t[c] = (c & 128) ? -v : v;
    }
    return t;
}();
namespace ISA_NS {
void pf_init() { pfc::tables_init(); }
void pf_job(const PfCpu::Job& J, const uint16_t* xq, const int8_t* xe, uint16_t* xg, float* h13, uint16_t* hq) {
    if (xe) {   // [dsv4-pf8-xqh] value = code * 2^e (exact in fp32), bf16 by truncation (exact: <= 4 significant bits)
        const uint8_t* pc = (const uint8_t*)xq;
        for (int i = 0; i < J.m; i++) {
            const uint8_t* cr = pc + (size_t)J.rows[i] * D;
            const int8_t* er = xe + (size_t)J.rows[i] * (D / 32);
            uint16_t* o = &xg[(size_t)i * D];
            for (int b = 0; b < D / 32; b++) {
                const float sc = std::ldexp(1.f, er[b]);
                for (int t = 0; t < 32; t++) {
                    const float v = g_e4m3[cr[b * 32 + t]] * sc;
                    uint32_t u;
                    memcpy(&u, &v, 4);
                    o[b * 32 + t] = (uint16_t)(u >> 16);
                }
            }
        }
    } else
        for (int i = 0; i < J.m; i++) memcpy(&xg[(size_t)i * D], xq + (size_t)J.rows[i] * D, (size_t)D * 2);
    pfc::expert((XFmt)J.fmt, J.base, J.m, xg, J.wt, h13, hq, J.y);
}
}  // namespace ISA_NS

}  // namespace dsv4
