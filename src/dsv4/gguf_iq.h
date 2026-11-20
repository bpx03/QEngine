#pragma once
// R16 [dsv4-gguf-iq]: GGUF i-quant expert formats (Maya-S's exact expert bytes) in the engine's per-matrix layout.
//
// Formats (ggml block definitions, QK_K = 256 weights per super-block, fp16 d first in every block):
//   IQ2_XXS  66 B: d | 8 x (4 grid indices u8 into a 256 x 8 grid, u32: 4 x 7 sign bits + 4-bit scale s)
//                  w = d * (0.5 + s) / 4 * grid * sign
//   IQ3_XXS  98 B: d | 64 grid indices u8 into a 256 x 4 grid | 8 x u32 (4 x 7 sign bits + 4-bit scale s)
//                  w = d * (0.5 + s) / 2 * grid * sign
//   IQ2_S    82 B: d | 32 low index bytes | 32 sign bytes | 8 qh (2 high index bits per entry) | 8 scales (2 nibbles)
//                  10-bit index into a 1024 x 8 grid; w = d * (0.5 + nibble) / 4 * grid * sign (nibble per 16 weights)
//   IQ3_S   110 B: d | 64 low index bytes | 8 qh (1 high bit per entry) | 32 sign bytes | 4 scales (2 nibbles)
//                  9-bit index into a 512 x 4 grid; w = d * (1 + 2 nibble) * grid * sign (nibble per 32 weights)
// 7-bit sign fields expand to 8 sign bits with even parity (bit 7 = popcount(bits 0..6) & 1).
// All products are exact in fp32 (d has 11 significant bits, the rest <= 11 more).
//
// Engine layout of one matrix [N][K] (lossless re-pack of the GGUF bytes, same byte count): the blocks without
// their d (codes, row-major, K / 256 per row, cbytes each) then all d as fp16 [N][K / 256] - the dsv4.h
// "codes then scales" convention, and the code bytes of a block start 16-byte aligned for IQ2_XXS / IQ3_XXS /
// IQ2_S (64 / 96 / 80 B) and 4-byte aligned for IQ3_S (108 B).
#include <cstdint>
#include <cstring>

#include "dsv4.h"
#include "gguf_grids.h"

namespace dsv4 {

// ggml type ids (gguf file format)
enum GgType : uint32_t {
    GG_F32 = 0, GG_F16 = 1, GG_Q8_0 = 8, GG_Q6_K = 14, GG_IQ2_XXS = 16, GG_IQ2_XS = 17, GG_IQ3_XXS = 18,
    GG_IQ3_S = 21, GG_IQ2_S = 22, GG_BF16 = 30
};
inline const char* gg_type_name(uint32_t t) {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1"; case 6: return "Q5_0";
        case 7: return "Q5_1"; case 8: return "Q8_0"; case 10: return "Q2_K"; case 11: return "Q3_K"; case 12: return "Q4_K";
        case 13: return "Q5_K"; case 14: return "Q6_K"; case 15: return "Q8_K"; case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 19: return "IQ1_S"; case 20: return "IQ4_NL";
        case 21: return "IQ3_S"; case 22: return "IQ2_S"; case 23: return "IQ4_XS"; case 24: return "I8"; case 25: return "I16";
        case 26: return "I32"; case 27: return "I64"; case 28: return "F64"; case 29: return "IQ1_M"; case 30: return "BF16";
        case 39: return "MXFP4";
    }
    return "?";
}
// (block weights, block bytes) of a ggml type; 0 when unknown here
inline void gg_type_block(uint32_t t, int& bw, int& bb) {
    bw = 256;
    switch (t) {
        case GG_F32: bw = 1; bb = 4; return;
        case GG_F16: case GG_BF16: bw = 1; bb = 2; return;
        case GG_Q8_0: bw = 32; bb = 34; return;
        case GG_Q6_K: bb = 210; return;
        case GG_IQ2_XXS: bb = 66; return;
        case GG_IQ2_XS: bb = 74; return;
        case GG_IQ3_XXS: bb = 98; return;
        case GG_IQ3_S: bb = 110; return;
        case GG_IQ2_S: bb = 82; return;
        case 2: bw = 32; bb = 18; return;
        case 3: bw = 32; bb = 20; return;
        case 6: bw = 32; bb = 22; return;
        case 7: bw = 32; bb = 24; return;
        case 10: bb = 84; return;
        case 11: bb = 110; return;
        case 12: bb = 144; return;
        case 13: bb = 176; return;
        case 20: bw = 32; bb = 18; return;
        case 23: bb = 136; return;
        case 29: bb = 56; return;
        case 19: bb = 50; return;
    }
    bw = 0;
    bb = 0;
}

// matrix format (XFmt 4..7, dsv4.h) <-> ggml type
inline XFmt gg_mat_fmt(uint32_t t) {
    return t == GG_IQ2_XXS ? XF_IQ2XXS : t == GG_IQ3_XXS ? XF_IQ3XXS : t == GG_IQ2_S ? XF_IQ2S : t == GG_IQ3_S ? XF_IQ3S : (XFmt)255;
}
inline uint32_t gg_type_of(XFmt mf) {
    return mf == XF_IQ2XXS ? GG_IQ2_XXS : mf == XF_IQ3XXS ? GG_IQ3_XXS : mf == XF_IQ2S ? GG_IQ2_S : GG_IQ3_S;
}

// ---------------------------------------------------------------- scalar dequant (reference; host and device)

__host__ __device__ inline float gg_h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        if (!m) x = s;
        else {   // subnormal: m * 2^-24
            float f = (float)m * 5.9604644775390625e-8f;
            return s ? -f : f;
        }
    } else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}
__host__ __device__ inline uint8_t gg_ksign(uint32_t v7) {   // 7 sign bits -> 8 with even parity
#ifdef __CUDA_ARCH__
    return (uint8_t)(v7 | ((__popc(v7) & 1) << 7));
#else
    return (uint8_t)(v7 | ((__builtin_popcount(v7) & 1) << 7));
#endif
}

// one super-block (code bytes c, d) -> 256 floats (host; the grids are host tables here)
inline void gg_deq_block(XFmt mf, const uint8_t* c, float d, float* y) {
    if (mf == XF_IQ2XXS) {
        for (int k = 0; k < 8; k++) {
            uint32_t aux;
            memcpy(&aux, c + 8 * k + 4, 4);
            const float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint64_t g = gg_iq2xxs_grid[c[8 * k + l]];
                const uint8_t sg = gg_ksign((aux >> (7 * l)) & 127);
                for (int j = 0; j < 8; j++)
                    y[32 * k + 8 * l + j] = db * (float)((g >> (8 * j)) & 255) * ((sg >> j) & 1 ? -1.f : 1.f);
            }
        }
    } else if (mf == XF_IQ3XXS) {
        for (int k = 0; k < 8; k++) {
            uint32_t aux;
            memcpy(&aux, c + 64 + 4 * k, 4);
            const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = gg_ksign((aux >> (7 * l)) & 127);
                const uint32_t g1 = gg_iq3xxs_grid[c[8 * k + 2 * l]], g2 = gg_iq3xxs_grid[c[8 * k + 2 * l + 1]];
                for (int j = 0; j < 4; j++) {
                    y[32 * k + 8 * l + j] = db * (float)((g1 >> (8 * j)) & 255) * ((sg >> j) & 1 ? -1.f : 1.f);
                    y[32 * k + 8 * l + 4 + j] = db * (float)((g2 >> (8 * j)) & 255) * ((sg >> (j + 4)) & 1 ? -1.f : 1.f);
                }
            }
        }
    } else if (mf == XF_IQ2S) {
        const uint8_t *qs = c, *sg = c + 32, *qh = c + 64, *sc = c + 72;
        for (int k = 0; k < 8; k++) {
            const float db0 = d * (0.5f + (float)(sc[k] & 15)) * 0.25f, db1 = d * (0.5f + (float)(sc[k] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint64_t g = gg_iq2s_grid[qs[4 * k + l] | (((qh[k] >> (2 * l)) & 3) << 8)];
                const uint8_t s = sg[4 * k + l];
                const float db = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; j++)
                    y[32 * k + 8 * l + j] = db * (float)((g >> (8 * j)) & 255) * ((s >> j) & 1 ? -1.f : 1.f);
            }
        }
    } else {   // XF_IQ3S
        const uint8_t *qs = c, *qh = c + 64, *sg = c + 72, *sc = c + 104;
        for (int k = 0; k < 8; k++) {
            const float db = d * (float)(1 + 2 * ((sc[k / 2] >> (4 * (k & 1))) & 15));
            for (int i = 0; i < 8; i++) {   // entry i of the 32: 4 weights
                const uint32_t g = gg_iq3s_grid[qs[8 * k + i] | (((qh[k] >> i) & 1) << 8)];
                const uint8_t s = sg[4 * k + i / 2];
                for (int j = 0; j < 4; j++)
                    y[32 * k + 4 * i + j] = db * (float)((g >> (8 * j)) & 255) * ((s >> (4 * (i & 1) + j)) & 1 ? -1.f : 1.f);
            }
        }
    }
}

// raw GGUF rows (blocks with d) [N][K] -> engine layout (codes then d); and back
inline void gg_repack(XFmt mf, const uint8_t* raw, int N, int K, uint8_t* out) {
    const int nb = K / 256, cb = gg_cbytes(mf);
    uint8_t* codes = out;
    uint8_t* ds = out + (size_t)N * nb * cb;
    for (size_t b = 0; b < (size_t)N * nb; b++) {
        memcpy(ds + 2 * b, raw + b * (cb + 2), 2);
        memcpy(codes + b * cb, raw + b * (cb + 2) + 2, cb);
    }
}
// matrix in engine layout -> fp32 rows [N][K]
inline void gg_deq_mat(XFmt mf, const uint8_t* q, const void* s, int N, int K, float* y) {
    const int nb = K / 256, cb = gg_cbytes(mf);
    const uint16_t* ds = (const uint16_t*)s;
    for (size_t b = 0; b < (size_t)N * nb; b++) gg_deq_block(mf, q + b * cb, gg_h2f(ds[b]), y + b * 256);
}

}  // namespace dsv4
