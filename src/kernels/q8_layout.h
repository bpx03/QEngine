#pragma once
// Q8: the INT8 layout of the 27B layer weights (opt-in INT8 checkpoints).
//
// One layout serves the decode GEMV (fp16 m16n8k16 fragments, XH activations) and the prefill GEMM (s8 m16n8k32
// fragments, int8 activations), so the weights are stored once:
//   q: [N/16][K/32][32 lanes][uint4]   (uint4 = one m16n8k32 s8 A fragment {a0, a1, a2, a3})
//   s: [N/16][K/128][16] bf16          (the W4 scale layout)
// For lane (g, t) = (lane / 4, lane % 4), 32-k sub-chunk c and k-step s in {0, 1} (k = 32 c + 16 s + [0, 16)):
//   word 2 s     (a0 / a2): row 16 r + g,     bytes i = 0..3 -> k = 32 c + 16 s + 2 t + {0, 8, 1, 9}[i]
//   word 2 s + 1 (a1 / a3): row 16 r + g + 8, same k
// bytes are the signed int8 weights (two's complement). The k order inside a word is the permutation the W4A8
// pre-pass (x_to_i8_kernel) applies to the activations (byte 4u + i of a 16-k half <- k 2u + {0, 8, 1, 9}[i]), so
// the s8 MMA takes the words as they are. The fp16 m16n8k16 fragment of k-step s takes pairs (2t, 2t+1) = bytes
// {0, 2} and (2t+8, 2t+9) = bytes {1, 3} of a word: one byte_perm each (q8_h2 in gemv.cu).
//
// Host + device helpers (no CUDA dependency here: tests/q8_layout_unit.cpp checks them on the CPU).
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q8_HD __host__ __device__ __forceinline__
#else
#define Q8_HD inline
#endif

// byte i (0..3) of a Q8 word -> k offset inside the 16-k step, relative to 2t
Q8_HD int q8_koff(int i) { return (i & 1) * 8 + (i >> 1); }

// position of weight (row, k) of an [N][K] matrix in the Q8 byte array
Q8_HD size_t q8_pos(int row, int k, int K) {
    const int r = row >> 4, rr = row & 15, g = rr & 7, hi = rr >> 3;
    const int c = k >> 5, s = (k >> 4) & 1, kk = k & 15;  // kk = 2t + koff
    const int b8 = kk >> 3, rem = kk & 7;                  // rem = 2t + (kk & 1)
    const int t = rem >> 1, i = (b8 & 1) | ((kk & 1) << 1);
    const int lane = g * 4 + t, word = 2 * s + hi;
    return (((size_t)r * (K >> 5) + c) * 32 + lane) * 16 + word * 4 + i;
}

inline size_t q8_qbytes(int n, int k) { return (size_t)n * k; }

// fp16 m16n8k16 fragment selector of a Q8 word (byte_perm with the 0x54 exponent byte at index 4): pair (2t, 2t+1)
// = bytes 0 / 2, pair (2t+8, 2t+9) = bytes 1 / 3
constexpr uint32_t kQ8SelLo = 0x4240u, kQ8SelHi = 0x4341u;

#if defined(__CUDACC__)
#include <cuda_fp16.h>
// Two signed bytes of a Q8 word -> fp16x2 w / 16, exact: byte_perm puts u = w + 128 (the word is pre-XORed with
// 0x80808080) under the exponent byte 0x54 (64 + u / 16), the subtraction of 72 leaves w / 16. One PRMT + one HSUB2
// per pair (the INT4 xh_dequant needs a shift, a LOP3 and the HSUB2).
__device__ __forceinline__ uint32_t q8_h2(uint32_t wx, uint32_t sel) {
    uint32_t v = __byte_perm(wx, 0x54u, sel);
    const __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __float2half2_rn(72.f));
    return *reinterpret_cast<const uint32_t*>(&h);
}
// fp16 A fragment (m16n8k16) of k-step s of a Q8 uint4 (two k-steps), entries w / 16
__device__ __forceinline__ void q8_frag_h(const uint4& wc, int s, uint32_t a[4]) {
    const uint32_t wa = (s ? wc.z : wc.x) ^ 0x80808080u, wb = (s ? wc.w : wc.y) ^ 0x80808080u;
    a[0] = q8_h2(wa, kQ8SelLo);
    a[1] = q8_h2(wb, kQ8SelLo);
    a[2] = q8_h2(wa, kQ8SelHi);
    a[3] = q8_h2(wb, kQ8SelHi);
}
#endif
