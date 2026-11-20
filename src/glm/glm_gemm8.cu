// Prefill dense projections on INT8 tensor cores (round 10, P-2): Y[t][n] (+)= sum_g sx[t][g] sw[n][g] (x_g . w_g)
// with the stored INT8 (or INT4) g128 weights used as they are (no dequantisation pass, no bf16 rounding of w * s) and
// the activations quantised per (token, 128-group). NP = 2 passes: x ~ s (q1 + q2 / 256) (the residual of the first
// int8 pass in a second int8 plane, ~15 bits of the group absmax, finer than the bf16 input); the group's int32 sum is
// (q1 . w) << 8 + q2 . w (|.| < 2^30), so one accumulator set serves both passes. NP = 1: one int8 pass (lossy).
// 128 x 128 tiles, 256 threads (2 x 4 warps of 64 x 32), one K-group (128) per stage, 2 stages: cp.async for the
// activations and INT8 weights, register-staged nibble expansion for INT4 weights; ldmatrix on XOR-swizzled rows.
#include "glm.h"
#include "glm_kernels.cuh"

namespace glm {

namespace {
__device__ __forceinline__ void mma8(int* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cpa16(void* sm, const void* gm, bool v) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gm), "r"(v ? 16 : 0));
}
__device__ __forceinline__ void cpa4(void* sm, const void* gm, bool v) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gm), "r"(v ? 4 : 0));
}
__device__ __forceinline__ void ldsm4(unsigned* r, const void* sm) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// 8 signed nibbles (low nibble = even k) -> 8 int8 in two words (k0..k3 | k4..k7)
__device__ __forceinline__ void snib2s8(unsigned w, unsigned& lo, unsigned& hi) {
    w ^= 0x88888888u;  // signed nibble v = (u ^ 8) - 8
    const unsigned e = w & 0x0F0F0F0Fu, o = (w >> 4) & 0x0F0F0F0Fu;
    const unsigned a = __byte_perm(e, o, 0x5140), b = __byte_perm(e, o, 0x7362);
    lo = (a + 0x78787878u) ^ 0x80808080u;
    hi = (b + 0x78787878u) ^ 0x80808080u;
}
}  // namespace

constexpr int G8M = 128, G8N = 128, G8K = 128, G8T = 256;
// swizzled byte offset of 16-byte chunk c of row r in a tile of rows of (cpr * 16) bytes
__device__ __forceinline__ int g8sw(int r, int c, int cpr) { return r * cpr * 16 + (((c & ~7) | ((c & 7) ^ (r & 7))) << 4); }

template <int NP, bool W4>
__global__ void __launch_bounds__(G8T, 1) k_gemm8(const int8_t* __restrict__ xq, const float* __restrict__ xs, int T,
                                                  const int8_t* __restrict__ wq, const uint8_t* __restrict__ wq4,
                                                  const float* __restrict__ ws, int N, int K, float* __restrict__ Y,
                                                  int ldy, bool acc) {
    extern __shared__ __align__(16) unsigned char sm[];
    constexpr int ACPR = G8K * NP / 16, BCPR = G8K / 16;  // 16-byte chunks per row
    constexpr int ABYTES = G8M * G8K * NP, BBYTES = G8N * G8K, SBYTES = ABYTES + BBYTES + 2 * 128 * 4;
    const int m0 = blockIdx.y * G8M, n0 = blockIdx.x * G8N, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp >> 2) * 64, wn = (wp & 3) * 32;
    const int KG = K / G8K;
    const size_t lda = (size_t)K * NP;
    auto As = [&](int b) { return sm + b * SBYTES; };
    auto Bs = [&](int b) { return sm + b * SBYTES + ABYTES; };
    auto Sx = [&](int b) { return reinterpret_cast<float*>(sm + b * SBYTES + ABYTES + BBYTES); };
    auto Sw = [&](int b) { return reinterpret_cast<float*>(sm + b * SBYTES + ABYTES + BBYTES + 512); };
    uint4 rb[2];  // W4: next group's nibbles (2 x 16 bytes per thread = 128 columns x 64 bytes)
    auto load = [&](int g, int b) {
        unsigned char* a = As(b);
#pragma unroll
        for (int u = 0; u < G8M * ACPR / G8T; u++) {
            const int i = tid + u * G8T, r = i / ACPR, c = i % ACPR, t = m0 + r;
            cpa16(a + g8sw(r, c, ACPR), xq + (size_t)min(t, T - 1) * lda + (size_t)g * G8K * NP + c * 16, t < T);
        }
        if constexpr (!W4) {
            unsigned char* bb = Bs(b);
#pragma unroll
            for (int u = 0; u < G8N * BCPR / G8T; u++) {
                const int i = tid + u * G8T, r = i / BCPR, c = i % BCPR, n = n0 + r;
                cpa16(bb + g8sw(r, c, BCPR), wq + (size_t)min(n, N - 1) * K + (size_t)g * G8K + c * 16, n < N);
            }
        } else {
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int i = tid + u * G8T, r = i / 4, c = i % 4, n = n0 + r;
                rb[u] = n < N ? __ldg(reinterpret_cast<const uint4*>(wq4 + (size_t)n * (K / 2) + (size_t)g * (G8K / 2) + c * 16))
                              : make_uint4(0, 0, 0, 0);
            }
        }
        if (tid < 128) {
            const int t = m0 + tid;
            cpa4(Sx(b) + tid, xs + (size_t)min(t, T - 1) * KG + g, t < T);
        } else {
            const int n = n0 + tid - 128;
            cpa4(Sw(b) + tid - 128, ws + (size_t)min(n, N - 1) * KG + g, n < N);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto store4 = [&](int b) {  // W4: expand the staged nibbles into int8 rows of the B tile
        if constexpr (W4) {
            unsigned char* bb = Bs(b);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int i = tid + u * G8T, r = i / 4, c = i % 4;  // 16 bytes of nibbles = k 32c .. 32c + 31
                const unsigned w[4] = {rb[u].x, rb[u].y, rb[u].z, rb[u].w};
                uint4 o0, o1;
                snib2s8(w[0], o0.x, o0.y); snib2s8(w[1], o0.z, o0.w);
                snib2s8(w[2], o1.x, o1.y); snib2s8(w[3], o1.z, o1.w);
                *reinterpret_cast<uint4*>(bb + g8sw(r, 2 * c, BCPR)) = o0;
                *reinterpret_cast<uint4*>(bb + g8sw(r, 2 * c + 1, BCPR)) = o1;
            }
        }
    };
    float f[4][4][4] = {};
    load(0, 0);
    store4(0);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    for (int g = 0; g < KG; g++) {
        const int b = g & 1;
        if (g + 1 < KG) load(g + 1, b ^ 1);
        const unsigned char* a = As(b);
        const unsigned char* bb = Bs(b);
        int c[4][4][4] = {};
#pragma unroll
        for (int p = 0; p < NP; p++) {
            if (p == 1) {
#pragma unroll
                for (int mi = 0; mi < 4; mi++)
#pragma unroll
                    for (int nj = 0; nj < 4; nj++)
#pragma unroll
                        for (int e = 0; e < 4; e++) c[mi][nj][e] <<= 8;
            }
#pragma unroll
            for (int ks = 0; ks < G8K; ks += 32) {
                unsigned af[4][4], bf[4][2];
#pragma unroll
                for (int mi = 0; mi < 4; mi++) {
                    const int r = wm + mi * 16 + (ln & 7) + ((ln >> 3) & 1) * 8;
                    const int ch = (p * G8K + ks) / 16 + (ln >> 4);
                    ldsm4(af[mi], a + g8sw(r, ch, ACPR));
                }
#pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const int r = wn + nh * 16 + (ln & 7) + (ln >> 4) * 8;
                    const int ch = ks / 16 + ((ln >> 3) & 1);
                    unsigned t4[4];
                    ldsm4(t4, bb + g8sw(r, ch, BCPR));
                    bf[2 * nh][0] = t4[0]; bf[2 * nh][1] = t4[1];
                    bf[2 * nh + 1][0] = t4[2]; bf[2 * nh + 1][1] = t4[3];
                }
#pragma unroll
                for (int mi = 0; mi < 4; mi++)
#pragma unroll
                    for (int nj = 0; nj < 4; nj++) mma8(c[mi][nj], af[mi], bf[nj]);
            }
        }
        const float* sx = Sx(b);
        const float* sw = Sw(b);
        constexpr float PS = NP == 2 ? 1.f / 256.f : 1.f;
#pragma unroll
        for (int mi = 0; mi < 4; mi++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, cc = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                    f[mi][nj][e] += (float)c[mi][nj][e] * (sx[r] * PS) * sw[cc];
                }
        if (g + 1 < KG) store4(b ^ 1);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int nj = 0; nj < 4; nj++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int t = m0 + wm + mi * 16 + (ln >> 2) + h * 8, n = n0 + wn + nj * 8 + (ln & 3) * 2;
                if (t >= T) continue;
                float* y = Y + (size_t)t * ldy + n;
                const float v0 = f[mi][nj][2 * h], v1 = f[mi][nj][2 * h + 1];
                if (n + 1 < N && !(ldy & 1)) {
                    float2 o = make_float2(v0, v1);
                    if (acc) { const float2 q = *reinterpret_cast<const float2*>(y); o.x += q.x; o.y += q.y; }
                    *reinterpret_cast<float2*>(y) = o;
                } else {
                    if (n < N) y[0] = acc ? y[0] + v0 : v0;
                    if (n + 1 < N) y[1] = acc ? y[1] + v1 : v1;
                }
            }
}

// x (bf16 rows, stride ldx) -> int8 per (token, 128-group): NP planes per group [q1 128 | q2 128], scales [T][K/128]
template <int NP>
__global__ void k_q8x(const bf16* __restrict__ x, int ldx, int K, int8_t* __restrict__ q, float* __restrict__ s) {
    const int g = blockIdx.x, t = blockIdx.y, ln = threadIdx.x;
    const bf16* r = x + (size_t)t * ldx + g * 128 + ln * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(r);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const bf162*>(&raw.x)),
                 b = __bfloat1622float2(*reinterpret_cast<const bf162*>(&raw.y));
    const float v[4] = {a.x, a.y, b.x, b.y};
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sc = m > 0.f ? m / 127.f : 1.f, inv = m > 0.f ? 127.f / m : 0.f;
    int q1[4];
#pragma unroll
    for (int j = 0; j < 4; j++) q1[j] = max(-127, min(127, __float2int_rn(v[j] * inv)));
    int8_t* o = q + ((size_t)t * (K / 128) + g) * 128 * NP + ln * 4;
    *reinterpret_cast<char4*>(o) = make_char4((char)q1[0], (char)q1[1], (char)q1[2], (char)q1[3]);
    if (NP == 2) {
        int q2[4];
#pragma unroll
        for (int j = 0; j < 4; j++) q2[j] = max(-127, min(127, __float2int_rn((v[j] - sc * (float)q1[j]) * inv * 256.f)));
        *reinterpret_cast<char4*>(o + 128) = make_char4((char)q2[0], (char)q2[1], (char)q2[2], (char)q2[3]);
    }
    if (ln == 0) s[(size_t)t * (K / 128) + g] = sc;
}

void gemm8_quant(const bf16* x, int ldx, int K, int T, int np, int8_t* q, float* s, cudaStream_t st) {
    if (np == 2) k_q8x<2><<<dim3(K / 128, T), 32, 0, st>>>(x, ldx, K, q, s);
    else k_q8x<1><<<dim3(K / 128, T), 32, 0, st>>>(x, ldx, K, q, s);
}

template <int NP, bool W4>
static void launch8(const int8_t* xq, const float* xs, int T, const W8& w, float* Y, int ldy, bool acc, cudaStream_t st) {
    constexpr int SB = G8M * G8K * NP + G8N * G8K + 2 * 128 * 4;
    static bool init = false;  // (per process: the attribute is per function, set on every device once is enough here)
    static int dev_done[8] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!dev_done[dev & 7]) {
        cudaFuncSetAttribute(k_gemm8<NP, W4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * SB);
        dev_done[dev & 7] = 1;
    }
    (void)init;
    dim3 grid((w.N + G8N - 1) / G8N, (T + G8M - 1) / G8M);
    k_gemm8<NP, W4><<<grid, G8T, 2 * SB, st>>>(xq, xs, T, w.q, w.q4, w.s, w.N, w.K, Y, ldy, acc);
}
void gemm8(const int8_t* xq, const float* xs, int T, const W8& w, float* Y, int ldy, bool acc, int np, cudaStream_t st) {
    if (np == 2) {
        if (w.q4) launch8<2, true>(xq, xs, T, w, Y, ldy, acc, st);
        else launch8<2, false>(xq, xs, T, w, Y, ldy, acc, st);
    } else {
        if (w.q4) launch8<1, true>(xq, xs, T, w, Y, ldy, acc, st);
        else launch8<1, false>(xq, xs, T, w, Y, ldy, acc, st);
    }
}

}  // namespace glm
