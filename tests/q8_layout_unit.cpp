// CPU check of the Q8 weight layout (src/kernels/q8_layout.h) against the MMA fragment definitions the kernels use.
// Build: g++ -std=c++20 -O2 -I src tests/q8_layout_unit.cpp -o /tmp/q8u && /tmp/q8u
// 1. s8 m16n8k32 (prefill W8A8): the uint4 of (tile, 32-k sub-chunk, lane) is {a0, a1, a2, a3} as is; with the
//    activations in the x_to_i8_kernel permutation, sum over lanes of the MMA = the exact dot products.
// 2. fp16 m16n8k16 (decode H16): byte_perm(word ^ 0x80808080, 0x54, kQ8SelLo / kQ8SelHi) - 72 = w / 16 for the
//    (2t, 2t+1) / (2t+8, 2t+9) pairs, with the XH activation positions of gemv_h16.cuh.
// 3. repack from compressed-tensors pack-quantized 8-bit (int32 [N][K/4], byte b of word j = k 4j + b, u = w + 128).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/q8_layout.h"

static int fails = 0;
#define EXPECT(c, ...)                         \
    do {                                       \
        if (!(c)) {                            \
            if (fails++ < 10) printf(__VA_ARGS__); \
        }                                      \
    } while (0)

static uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {  // PTX prmt (default mode)
    const uint64_t v = ((uint64_t)y << 32) | x;
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) {
        const int sel = (s >> (4 * i)) & 7;
        r |= (uint32_t)((v >> (8 * sel)) & 0xFF) << (8 * i);
    }
    return r;
}
static float half_to_float(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp((float)m, -24) : std::ldexp(1.f + m / 1024.f, e - 15);
    return (h & 0x8000) ? -v : v;
}
// gemv_h16.cuh xh_pos: column of k inside the XH row
static int xh_pos(int k) { return (k & ~63) | (k & 1) | ((k >> 2) & 14) | ((k & 6) << 3); }

int main() {
    const int N = 64, K = 256;
    std::mt19937 rng(7);
    std::vector<int8_t> W((size_t)N * K);
    for (auto& w : W) w = (int8_t)((int)(rng() % 256) - 128);
    std::vector<int> X(K);
    for (auto& x : X) x = (int)(rng() % 255) - 127;
    std::vector<uint8_t> q(q8_qbytes(N, K), 0xEE);
    std::vector<int> hit(q.size(), 0);
    for (int r = 0; r < N; r++)
        for (int k = 0; k < K; k++) {
            const size_t p = q8_pos(r, k, K);
            hit[p]++;
            q[p] = (uint8_t)W[(size_t)r * K + k];
        }
    for (size_t i = 0; i < hit.size(); i++) EXPECT(hit[i] == 1, "q8_pos not a bijection at %zu (%d)\n", i, hit[i]);

    // 1. s8 m16n8k32: activation bytes in the pre-pass permutation (per 16-k half: phys 4u + i <- k 2u + koff(i))
    std::vector<int8_t> xp(K);
    for (int h = 0; h < K / 16; h++)
        for (int u = 0; u < 4; u++)
            for (int i = 0; i < 4; i++) xp[h * 16 + 4 * u + i] = (int8_t)X[h * 16 + 2 * u + q8_koff(i)];
    for (int r = 0; r < N / 16; r++) {
        long long acc[16] = {};
        for (int c = 0; c < K / 32; c++)
            for (int lane = 0; lane < 32; lane++) {
                const int g = lane >> 2, t = lane & 3;
                const uint8_t* fr = &q[(((size_t)r * (K / 32) + c) * 32 + lane) * 16];
                // a0: row g, phys k 4t..4t+3 (first 16); a1: row g+8; a2: row g, phys 16 + 4t..; a3: row g+8
                for (int a = 0; a < 4; a++) {
                    const int row = g + (a & 1) * 8, ph = c * 32 + (a >> 1) * 16 + 4 * t;
                    for (int i = 0; i < 4; i++) acc[row] += (long long)(int8_t)fr[a * 4 + i] * xp[ph + i];
                }
            }
        for (int rr = 0; rr < 16; rr++) {
            long long ref = 0;
            for (int k = 0; k < K; k++) ref += (long long)W[(size_t)(r * 16 + rr) * K + k] * X[k];
            EXPECT(acc[rr] == ref, "s8 frag tile %d row %d: %lld vs %lld\n", r, rr, acc[rr], ref);
        }
    }

    // 2. fp16 m16n8k16 via byte_perm, B from the XH layout (lane t reads halves [16t, 16t + 16) of a 64-k chunk:
    //    k-step j -> b0 = halves 16t + 4j + {0, 1}, b1 = 16t + 4j + {2, 3})
    std::vector<float> xh(K);
    for (int k = 0; k < K; k++) xh[xh_pos(k)] = (float)X[k];
    for (int r = 0; r < N / 16; r++) {
        double acc[16] = {};
        for (int ch = 0; ch < K / 64; ch++)
            for (int lane = 0; lane < 32; lane++) {
                const int g = lane >> 2, t = lane & 3;
                for (int j = 0; j < 4; j++) {
                    const int c = 2 * ch + (j >> 1), s = j & 1;
                    const uint8_t* fr = &q[(((size_t)r * (K / 32) + c) * 32 + lane) * 16];
                    uint32_t wa, wb;
                    memcpy(&wa, fr + (2 * s) * 4, 4);
                    memcpy(&wb, fr + (2 * s + 1) * 4, 4);
                    float a[4][2];  // a0 row g (2t,2t+1), a1 row g+8, a2 row g (2t+8,..), a3 row g+8
                    const uint32_t src[4] = {wa, wb, wa, wb}, sel[4] = {kQ8SelLo, kQ8SelLo, kQ8SelHi, kQ8SelHi};
                    for (int f = 0; f < 4; f++) {
                        const uint32_t v = byte_perm(src[f] ^ 0x80808080u, 0x54u, sel[f]);
                        a[f][0] = half_to_float((uint16_t)(v & 0xFFFF)) - 72.f;
                        a[f][1] = half_to_float((uint16_t)(v >> 16)) - 72.f;
                    }
                    const float* xb = &xh[ch * 64 + 16 * t + 4 * j];
                    // row g: a0 . b0 + a2 . b1 ; row g+8: a1 . b0 + a3 . b1  (b0 = k 2t, 2t+1; b1 = 2t+8, 2t+9)
                    acc[g] += 16.0 * (a[0][0] * xb[0] + a[0][1] * xb[1] + a[2][0] * xb[2] + a[2][1] * xb[3]);
                    acc[g + 8] += 16.0 * (a[1][0] * xb[0] + a[1][1] * xb[1] + a[3][0] * xb[2] + a[3][1] * xb[3]);
                }
            }
        for (int rr = 0; rr < 16; rr++) {
            long long ref = 0;
            for (int k = 0; k < K; k++) ref += (long long)W[(size_t)(r * 16 + rr) * K + k] * X[k];
            EXPECT(acc[rr] == (double)ref, "f16 frag tile %d row %d: %.1f vs %lld\n", r, rr, acc[rr], ref);
        }
    }
    // the decode is exact for every byte value
    for (int b = 0; b < 256; b++) {
        const uint32_t v = byte_perm((uint32_t)b ^ 0x80u, 0x54u, kQ8SelLo);
        const float f = half_to_float((uint16_t)(v & 0xFFFF)) - 72.f;
        EXPECT(f * 16.f == (float)(int8_t)b, "decode byte %d -> %f\n", b, f);
    }

    // 3. compressed-tensors source -> Q8 (the repack_q8_kernel index rule)
    std::vector<int32_t> ct((size_t)N * K / 4);
    for (int r = 0; r < N; r++)
        for (int k = 0; k < K; k++) {
            const uint32_t u = (uint32_t)(W[(size_t)r * K + k] + 128);
            reinterpret_cast<uint32_t&>(ct[(size_t)r * (K / 4) + k / 4]) |= u << (8 * (k & 3));
        }
    for (int r = 0; r < N; r++)
        for (int k = 0; k < K; k++) {
            const uint32_t u = ((uint32_t)ct[(size_t)r * (K / 4) + k / 4] >> (8 * (k & 3))) & 0xFF;
            EXPECT((uint8_t)(u ^ 0x80u) == q[q8_pos(r, k, K)], "ct repack r %d k %d\n", r, k);
        }
    printf("q8_layout_unit: %s (%d failures)\n", fails ? "FAIL" : "ok", fails);
    return fails ? 1 : 0;
}
