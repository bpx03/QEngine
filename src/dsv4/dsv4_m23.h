#pragma once
// R16: load-time conversion of checkpoint FP4 experts to the Maya-budget format XF_M23 (w1 / w3 cb2g128, w2 cb3g128),
// the same rule as tools/dsv4/dsv4_quant.py quant_codes + dsv4_pack.py pack_cb: per 128-group absmax normalisation,
// nearest codebook level (sorted levels, midpoints, ties to the lower level as torch.searchsorted), least-squares
// scale refit, fp16 scale. CPU only (host code), one matrix per call.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "dsv4.h"

namespace dsv4 {

// fp32 -> fp16 bits, round to nearest even (as torch .to(float16)); normals, subnormals, overflow to inf
inline uint16_t m23_f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7fffffffu;
    if (x >= 0x7f800000u) return (uint16_t)(sign | 0x7c00u | (x > 0x7f800000u ? 0x200u : 0u));
    if (x >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   // >= 65520 rounds to inf
    if (x < 0x38800000u) {                                      // below the smallest normal half: subnormal
        if (x < 0x33000000u) return (uint16_t)sign;             // < 2^-25: rounds to 0
        const uint32_t m = (x & 0x7fffffu) | 0x800000u;
        const int sh = 126 - (int)(x >> 23);                    // 14 + (113 - exp) shifts
        const uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        return (uint16_t)(sign | (r + (rem > half || (rem == half && (r & 1)))));
    }
    const uint32_t r = ((x - 0x38000000u) >> 13), rem = x & 0x1fffu;
    return (uint16_t)(sign | (r + (rem > 0x1000u || (rem == 0x1000u && (r & 1)))));
}

inline float m23_e8m0(uint8_t e) {
    uint32_t u = (uint32_t)e << 23;
    float f;
    memcpy(&f, &u, 4);
    return e ? f : ldexpf(1.f, -127);
}

// q4 [N][K/2] FP4 e2m1 (lo nibble = element 2j), s8 [N][K/32] UE8M0 -> out: codes [N][K/128][gb] then fp16 scales
// [N][K/128] (gb = 48 for 3 bits, 32 for 2 bits)
inline void m23_quant_mat(const uint8_t* q4, const uint8_t* s8, int N, int K, int bits, const float* cb, uint8_t* out) {
    static const float lut[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const int L = 1 << bits, G = K / 128, gb = bits == 3 ? 48 : 32;
    float mid[7];
    for (int i = 0; i + 1 < L; i++) mid[i] = (cb[i + 1] + cb[i]) / 2.f;
    uint8_t* codes = out;
    uint16_t* scs = (uint16_t*)(out + (size_t)N * G * gb);
    float x[128];
    uint8_t a[128];
    for (int r = 0; r < N; r++) {
        const uint8_t* qr = q4 + (size_t)r * (K / 2);
        const uint8_t* sr = s8 + (size_t)r * (K / 32);
        for (int g = 0; g < G; g++) {
            float am = 0.f;
            for (int k = 0; k < 128; k++) {
                const int c = g * 128 + k;
                const uint8_t b = qr[c >> 1];
                x[k] = lut[(c & 1) ? (b >> 4) : (b & 15)] * m23_e8m0(sr[c >> 5]);
                am = fmaxf(am, fabsf(x[k]));
            }
            am = fmaxf(am, 1e-30f);
            float sxq = 0.f, sqq = 0.f;
            for (int k = 0; k < 128; k++) {
                const float xn = x[k] / am;
                int idx = 0;
                while (idx < L - 1 && mid[idx] < xn) idx++;   // searchsorted (left) on the midpoints
                a[k] = (uint8_t)idx;
                sxq += x[k] * cb[idx];
                sqq += cb[idx] * cb[idx];
            }
            const float s = sxq / fmaxf(sqq, 1e-30f);
            scs[(size_t)r * G + g] = m23_f2h(s);
            uint8_t* p = codes + ((size_t)r * G + g) * gb;
            for (int j = 0; j < 32; j++)
                p[j] = (uint8_t)((a[j] & 3) | ((a[j + 32] & 3) << 2) | ((a[j + 64] & 3) << 4) | ((a[j + 96] & 3) << 6));
            if (bits == 3)
                for (int j = 0; j < 16; j++) {
                    uint8_t h = 0;
                    for (int t = 0; t < 8; t++) h |= (uint8_t)(((a[8 * j + t] >> 2) & 1) << t);
                    p[32 + j] = h;
                }
        }
    }
}

// one expert: w[i] / s[i] = the checkpoint's FP4 bytes / UE8M0 scales of w1, w3, w2 -> dst (expert_bytes(XF_M23))
inline void m23_expert(const uint8_t* const w[3], const uint8_t* const s[3], const float* cb3, const float* cb2,
                       uint8_t* dst) {
    MatView mv[3];
    expert_views(XF_M23, dst, mv);
    const int Nm[3] = {INTER, INTER, D}, Km[3] = {D, D, INTER};
    for (int i = 0; i < 3; i++) {
        const XFmt mf = mat_fmt(XF_M23, i);
        m23_quant_mat(w[i], s[i], Nm[i], Km[i], cb_bits(mf), mf == XF_CB3 ? cb3 : cb2, (uint8_t*)mv[i].q);
    }
}

}  // namespace dsv4
