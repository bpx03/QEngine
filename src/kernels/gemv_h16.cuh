#pragma once
// fp16-accumulate INT4 GEMV for decoding (QE_GEMV_H16=1): one numeric scheme for every row count so
// that a sequence decoded in a batch gets bit-identical results to the same sequence decoded alone.
//
// Activations arrive in the "XH" layout written by their producer kernels (norm epilogues, attention /
// DeltaNet outputs, the GLU epilogue): row-major [m][ldx] fp16 holding x / 16 (an exact power-of-two
// scale; |x| is clamped to 2047 first so that a 64-k chunk partial sum, <= 64 * 8 * 128, stays below
// the fp16 maximum), with the columns of every 64-k chunk permuted into mma B-fragment order: lane t of
// the quad that holds token g reads halves [16t, 16t + 16) of the chunk, i.e. for k-step j = 0..3 the
// values at k = 16j + {2t, 2t+1, 2t+8, 2t+9}. Row offsets and any K that is a multiple of 64 work.
//
// Math (identical in every kernel - this header is the only definition): weights as exact integers
// (u - 8) in fp16, mma.m16n8k16 f16 x f16 -> f16 over the 4 k-steps of a chunk, then the chunk partial
// is added to fp32 accumulators scaled by (group scale * 16); the K slices (pick_warps, shape-only)
// are reduced in a fixed order exactly as in the bf16/fp32 kernels.
#include <cuda_fp16.h>

#include "../common.h"

constexpr float kXhScale = 16.f;
constexpr float kXhClamp = 2047.f;

// column of k within its row: inside the chunk, k bits (b0..b5) -> p = b0 | b3<<1 | b4<<2 | b5<<3 | b1<<4 | b2<<5
// (p = 16 t + 4 j + e with t = b2b1, j = b5b4, e = b3b0)
__device__ __forceinline__ int xh_pos(int k) {
    return (k & ~63) | (k & 1) | ((k >> 2) & 14) | ((k & 6) << 3);
}
__device__ __forceinline__ __half to_xh(float v) {
    return __float2half_rn(fminf(fmaxf(v, -kXhClamp), kXhClamp) * (1.f / kXhScale));
}
// value v of column k of token row `row` (row = xh + token * ldx)
__device__ __forceinline__ void store_xh(__half* row, int k, float v) { row[xh_pos(k)] = to_xh(v); }
// four consecutive columns k..k+3 (k % 4 == 0): two adjacent pairs in the XH layout
__device__ __forceinline__ void store_xh4(__half* row, int k, float a, float b, float c, float d) {
    const int p = xh_pos(k);
    *reinterpret_cast<__half2*>(row + p) = __halves2half2(to_xh(a), to_xh(b));
    *reinterpret_cast<__half2*>(row + p + 16) = __halves2half2(to_xh(c), to_xh(d));
}

__device__ __forceinline__ void xh_mma(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// packed nibbles u = w + 8 -> four fp16x2 A-fragment registers holding the exact integers w
__device__ __forceinline__ void xh_dequant(uint32_t q, uint32_t a[4]) {
    const __half2 bias = __halves2half2(__float2half(1032.f), __float2half(1032.f));
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t v = ((q >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
        __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), bias);
        a[j] = *reinterpret_cast<uint32_t*>(&h);
    }
}
// chunk partial h (fp16 C fragment) -> fp32 accumulators; sg / sg8 = group scale * kXhScale of rows
// g / g + 8 (exact: power of two)
__device__ __forceinline__ void xh_flush(float acc[4], uint32_t h[2], float sg, float sg8) {
    const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&h[0]));
    const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&h[1]));
    acc[0] = __fmaf_rn(lo.x, sg, acc[0]);
    acc[1] = __fmaf_rn(lo.y, sg, acc[1]);
    acc[2] = __fmaf_rn(hi.x, sg8, acc[2]);
    acc[3] = __fmaf_rn(hi.y, sg8, acc[3]);
    h[0] = h[1] = 0u;
}
__device__ __forceinline__ float xh_scale(bf16 s) { return __bfloat162float(s) * kXhScale; }
