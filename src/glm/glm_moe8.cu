// Prefill routed experts on INT8 tensor cores (W4A8): per expert, gate/up with the rows gathered from the int8
// activations (scale per (token, 128-group)) and the INT4 weights expanded to int8 in shared memory (scale per
// (128-group, column)); the glu and the int8 quantisation of h (scale per (row, 64 columns)) in the epilogue; down with
// the weighted scatter-add into part in the epilogue (experts run in order: every part element gets one add per expert,
// deterministic). mma.sync m16n8k32 s8 x s8 -> s32, int32 sums rescaled per K-group into fp32.
#include <atomic>
#include "glm.h"
#include "glm_kernels.cuh"

namespace glm {

__device__ __forceinline__ void mma_s8(int* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 8 nibbles (k0..k7 of one column) -> 8 int8 (q - 8) in two words
__device__ __forceinline__ void nib2s8(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned e = w & 0x0F0F0F0Fu, o = (w >> 4) & 0x0F0F0F0Fu;  // bytes: k0 k2 k4 k6 / k1 k3 k5 k7
    // interleave to k0 k1 k2 k3 | k4 k5 k6 k7, then subtract 8 per byte (values 0..15 -> -8..7, no borrow across bytes
    // since we add 0x78 and xor 0x80: q - 8 = (q + 120) ^ 0x80 for q in 0..15)
    const unsigned a = __byte_perm(e, o, 0x5140), b = __byte_perm(e, o, 0x7362);
    lo = (a + 0x78787878u) ^ 0x80808080u;
    hi = (b + 0x78787878u) ^ 0x80808080u;
}

constexpr int QM = 64, QN = 64, QK = 128, QPAD = 16;

// x (bf16 rows) -> int8 per (token, 128-group) + scales [T][32]
__global__ void k_q128(const bf16* x, int8_t* q, float* s) {
    const int g = blockIdx.x, t = blockIdx.y, ln = threadIdx.x;
    const bf16* r = x + (size_t)t * H + g * 128 + ln * 4;
    float v[4], m = 0.f;
    for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(r[j]); m = fmaxf(m, fabsf(v[j])); }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
    char4 c;
    c.x = (char)__float2int_rn(v[0] * inv); c.y = (char)__float2int_rn(v[1] * inv);
    c.z = (char)__float2int_rn(v[2] * inv); c.w = (char)__float2int_rn(v[3] * inv);
    *reinterpret_cast<char4*>(q + (size_t)t * H + g * 128 + ln * 4) = c;
    if (ln == 0) s[(size_t)t * (H / 128) + g] = sc;
}
void moe8_quant(const bf16* x, int T, int8_t* q, float* s, cudaStream_t st) {
    k_q128<<<dim3(H / 128, T), 32, 0, st>>>(x, q, s);
}

// gate/up of one expert: grid (ceil(n / 64), ni / 64), 256 threads = 2 (m) x 4 (n) warps of 32 x 16; the next K-group's
// global data is loaded into registers while the current one runs on the tensor cores
constexpr int GT = 256;
__device__ __forceinline__ void moe_gu8_body(const int8_t* xq, const float* xs, const int* rows, int n, const XPtr& w,
                                             int ni, int8_t* hq, float* hs) {
    __shared__ __align__(16) int8_t As[QM][QK + QPAD];
    __shared__ __align__(16) int8_t Bg[QN][QK + QPAD], Bu[QN][QK + QPAD];
    __shared__ float sxa[QM], swg[QN], swu[QN];
    __shared__ int srow[QM];
    const int m0 = blockIdx.x * QM, n0 = blockIdx.y * QN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp >> 2) * 32, wn = (wp & 3) * 16;
    if (tid < QM) srow[tid] = m0 + tid < n ? rows[m0 + tid] : -1;
    float fg[2][2][4] = {}, fu[2][2][4] = {};
    __syncthreads();
    // per thread: 2 A chunks (16 B), 4 B words of each matrix
    uint4 ra[2];
    unsigned rg[4], ru[4];
    float rxa = 0.f, rsg = 0.f, rsu = 0.f;
    auto gload = [&](int g) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (QK / 16), c = (i % (QK / 16)) * 16;
            const int row = srow[r];
            ra[u] = row >= 0 ? *reinterpret_cast<const uint4*>(xq + (size_t)row * H + g * QK + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            rg[u] = (unsigned)__ldg(w.gq + (size_t)(g * (QK / 8) + kk) * w.ld + n0 + c);
            ru[u] = (unsigned)__ldg(w.uq + (size_t)(g * (QK / 8) + kk) * w.ld + n0 + c);
        }
        if (tid < QM) rxa = srow[tid] >= 0 ? xs[(size_t)srow[tid] * (H / 128) + g] : 0.f;
        if (tid < QN) {
            rsg = __half2float(w.gs[(size_t)g * w.ld + n0 + tid]);
            rsu = __half2float(w.us[(size_t)g * w.ld + n0 + tid]);
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (QK / 16), c = (i % (QK / 16)) * 16;
            *reinterpret_cast<uint4*>(&As[r][c]) = ra[u];
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            unsigned lo, hi;
            nib2s8(rg[u], lo, hi);
            *reinterpret_cast<uint2*>(&Bg[c][kk * 8]) = make_uint2(lo, hi);
            nib2s8(ru[u], lo, hi);
            *reinterpret_cast<uint2*>(&Bu[c][kk * 8]) = make_uint2(lo, hi);
        }
        if (tid < QM) sxa[tid] = rxa;
        if (tid < QN) { swg[tid] = rsg; swu[tid] = rsu; }
    };
    gload(0);
    for (int g = 0; g < H / QK; g++) {
        sstore();
        __syncthreads();
        if (g + 1 < H / QK) gload(g + 1);  // in flight while the tensor cores run
        int ag[2][2][4] = {}, au[2][2][4] = {};
#pragma unroll
        for (int ks = 0; ks < QK; ks += 32) {
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r = wm + mi * 16 + (ln >> 2), k = ks + (ln & 3) * 4;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&As[r][k]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][k]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&As[r][k + 16]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][k + 16]);
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), k = ks + (ln & 3) * 4;
                unsigned bg[2], bu[2];
                bg[0] = *reinterpret_cast<const unsigned*>(&Bg[c][k]);
                bg[1] = *reinterpret_cast<const unsigned*>(&Bg[c][k + 16]);
                bu[0] = *reinterpret_cast<const unsigned*>(&Bu[c][k]);
                bu[1] = *reinterpret_cast<const unsigned*>(&Bu[c][k + 16]);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    mma_s8(ag[mi][nj], a[mi], bg);
                    mma_s8(au[mi][nj], a[mi], bu);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 2; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                    fg[mi][nj][e] += (float)ag[mi][nj][e] * sxa[r] * swg[c];
                    fu[mi][nj][e] += (float)au[mi][nj][e] * sxa[r] * swu[c];
                }
        __syncthreads();
    }
    // glu, then int8 per (row, these 64 columns)
    float* hsm = reinterpret_cast<float*>(&Bg[0][0]);  // [64][64] floats = 16 KB (Bg + Bu are 18 KB)
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int nj = 0; nj < 2; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                const float gv = fminf(fg[mi][nj][e], CLAMP), uv = fminf(fmaxf(fu[mi][nj][e], -CLAMP), CLAMP);
                hsm[r * QN + c] = gv / (1.f + __expf(-gv)) * uv;
            }
    __syncthreads();
    {   // 4 threads per row, 16 columns each
        const int r = tid >> 2, q4 = tid & 3;
        float m = 0.f;
        for (int c = 0; c < 16; c++) m = fmaxf(m, fabsf(hsm[r * QN + q4 * 16 + c]));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
        if (m0 + r < n) {
            int8_t* o = hq + (size_t)(m0 + r) * ni + n0 + q4 * 16;
            for (int c = 0; c < 16; c++) o[c] = (int8_t)__float2int_rn(hsm[r * QN + q4 * 16 + c] * inv);
            if (q4 == 0) hs[(size_t)(m0 + r) * (ni / 64) + n0 / 64] = sc;
        }
    }
}

__global__ void __launch_bounds__(GT, 2) k_moe_gu8(const int8_t* xq, const float* xs, const int* rows, int n, XPtr w, int ni,
                                                   int8_t* hq, float* hs) {
    moe_gu8_body(xq, xs, rows, n, w, ni, hq, hs);
}
// round 10: gate/up of a group of experts in one launch (blockIdx.z = job; h rows at the job's row-list offset)
__global__ void __launch_bounds__(GT, 2) k_moe_gu8b(const int8_t* xq, const float* xs, const int* rows, const GuJob* jobs,
                                                    int ni, int8_t* hq, float* hs) {
    const GuJob& j = jobs[blockIdx.z];
    if ((int)blockIdx.x * QM >= j.n) return;
    moe_gu8_body(xq, xs, rows + j.off, j.n, j.w, ni, hq + (size_t)j.off * ni, hs + (size_t)j.off * (ni / 64));
}
void moe8_gu_batch(const int8_t* xq, const float* xs, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni,
                   int8_t* hq, float* hs, cudaStream_t st) {
    if (!njobs || !maxn) return;
    k_moe_gu8b<<<dim3((maxn + QM - 1) / QM, ni / QN, njobs), GT, 0, st>>>(xq, xs, rows, jobs, ni, hq, hs);
}

// down of one expert: grid (ceil(n / 64), 4096 / 64), 256 threads = 2 x 4 warps of 32 x 16; K = ni in 64-groups (the
// slot's down scales are per 64 rows), next group prefetched into registers
__global__ void __launch_bounds__(GT, 2) k_moe_dn8(const int8_t* hq, const float* hs, const int* rows, const float* wts, int n,
                                                   XPtr w, int ni, float* part) {
    __shared__ __align__(16) int8_t As[QM][64 + QPAD];
    __shared__ __align__(16) int8_t Bs[QN][64 + QPAD];
    __shared__ float sxa[QM], sw[QN];
    const int m0 = blockIdx.x * QM, n0 = blockIdx.y * QN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp >> 2) * 32, wn = (wp & 3) * 16;
    float f[2][2][4] = {};
    uint4 ra;
    unsigned rb[2];
    float rxa = 0.f, rsw = 0.f;
    auto gload = [&](int g) {
        {
            const int r = tid / 4, c = (tid % 4) * 16;
            ra = m0 + r < n ? *reinterpret_cast<const uint4*>(hq + (size_t)(m0 + r) * ni + g * 64 + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            rb[u] = (unsigned)__ldg(w.dq + (size_t)(w.r0 / 8 + g * 8 + kk) * H + n0 + c);
        }
        if (tid < QM) rxa = m0 + tid < n ? hs[(size_t)(m0 + tid) * (ni / 64) + g] : 0.f;
        if (tid < QN) rsw = __half2float(w.ds[(size_t)((w.r0 + g * 64) / 64) * H + n0 + tid]);
    };
    gload(0);
    for (int g = 0; g < ni / 64; g++) {
        {
            const int r = tid / 4, c = (tid % 4) * 16;
            *reinterpret_cast<uint4*>(&As[r][c]) = ra;
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            unsigned lo, hi;
            nib2s8(rb[u], lo, hi);
            *reinterpret_cast<uint2*>(&Bs[c][kk * 8]) = make_uint2(lo, hi);
        }
        if (tid < QM) sxa[tid] = rxa;
        if (tid < QN) sw[tid] = rsw;
        __syncthreads();
        if (g + 1 < ni / 64) gload(g + 1);
        int acc[2][2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 64; ks += 32) {
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r = wm + mi * 16 + (ln >> 2), k = ks + (ln & 3) * 4;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&As[r][k]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][k]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&As[r][k + 16]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][k + 16]);
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), k = ks + (ln & 3) * 4;
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(&Bs[c][k]);
                b[1] = *reinterpret_cast<const unsigned*>(&Bs[c][k + 16]);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) mma_s8(acc[mi][nj], a[mi], b);
            }
        }
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 2; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                    f[mi][nj][e] += (float)acc[mi][nj][e] * sxa[r] * sw[c];
                }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8;
            if (m0 + r >= n) continue;
            const float wr = wts[m0 + r];
            float* o = part + (size_t)rows[m0 + r] * H + n0 + wn + (ln & 3) * 2 + (e & 1);
#pragma unroll
            for (int nj = 0; nj < 2; nj++) o[nj * 8] += wr * f[mi][nj][e];
        }
}

// ---------------------------------------------------------------- fp16 core (fp16 inputs and accumulation, as the
// cuBLAS COMPUTE_16F path): same fusions (gathered A rows, W4 expanded in shared memory with the weight scale folded
// in, glu in the gate/up epilogue, weighted scatter in the down epilogue, no separate dequantisation)
__device__ __forceinline__ void mma_f16(unsigned* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 8 nibbles (k0..k7) times a scale -> 8 fp16 in 4 words: each nibble pair placed under the fp16 exponent of 1024
// (prmt + or), 1032 subtracted (exact q - 8), one multiply by the scale (one rounding, as the dequantisation path)
__device__ __forceinline__ uint4 nib2h(unsigned w, __half sc) {
    const __half2 s2 = __half2half2(sc), off = __float2half2_rn(1032.f);
    const unsigned e = w & 0x0F0F0F0Fu, o = (w >> 4) & 0x0F0F0F0Fu;
    const unsigned p0 = __byte_perm(e, o, 0x5140), p1 = __byte_perm(e, o, 0x7362);  // k0 k1 k2 k3 | k4 k5 k6 k7
    unsigned r[4];
    const unsigned src[4] = {__byte_perm(p0, 0u, 0x4140), __byte_perm(p0, 0u, 0x4342), __byte_perm(p1, 0u, 0x4140),
                             __byte_perm(p1, 0u, 0x4342)};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned u = src[j] | 0x64006400u;
        __half2 h = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&u), off), s2);
        r[j] = *reinterpret_cast<unsigned*>(&h);
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}
constexpr int HK = 64, HPAD = 8;  // K per stage (halves), row padding
__device__ __forceinline__ void gu16_body(const bf16* xb, const int* rows, int n, const XPtr& w, int ni, __half* hb, int bx, int by) {
    __shared__ __align__(16) __half As[QM][HK + HPAD];
    __shared__ __align__(16) __half Bg[QN][HK + HPAD], Bu[QN][HK + HPAD];
    __shared__ int srow[QM];
    const int m0 = bx * QM, n0 = by * QN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp >> 2) * 32, wn = (wp & 3) * 16;
    if (tid < QM) srow[tid] = m0 + tid < n ? rows[m0 + tid] : -1;
    unsigned cg[2][2][2] = {}, cu[2][2][2] = {};
    __syncthreads();
    uint4 ra[2];
    unsigned rg[2], ru[2];
    __half sg[2], su[2];
    auto gload = [&](int st) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (HK / 8), c = (i % (HK / 8)) * 8;
            const int row = srow[r];
            ra[u] = row >= 0 ? *reinterpret_cast<const uint4*>(xb + (size_t)row * H + st * HK + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;  // 8 packed rows x 64 columns
            const size_t wr = (size_t)(st * (HK / 8) + kk) * w.ld + n0 + c;
            rg[u] = (unsigned)__ldg(w.gq + wr);
            ru[u] = (unsigned)__ldg(w.uq + wr);
            const size_t sr = (size_t)((st * HK) / 128) * w.ld + n0 + c;
            sg[u] = w.gs[sr];
            su[u] = w.us[sr];
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (HK / 8), c = (i % (HK / 8)) * 8;
            const bf16* b = reinterpret_cast<const bf16*>(&ra[u]);
            __align__(16) __half h[8];
#pragma unroll
            for (int j = 0; j < 8; j++) h[j] = __float2half(__bfloat162float(b[j]));
            *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<uint4*>(h);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            *reinterpret_cast<uint4*>(&Bg[c][kk * 8]) = nib2h(rg[u], sg[u]);
            *reinterpret_cast<uint4*>(&Bu[c][kk * 8]) = nib2h(ru[u], su[u]);
        }
    };
    gload(0);
    for (int st = 0; st < H / HK; st++) {
        sstore();
        __syncthreads();
        if (st + 1 < H / HK) gload(st + 1);
#pragma unroll
        for (int ks = 0; ks < HK; ks += 16) {
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r = wm + mi * 16 + (ln >> 2), k = ks + (ln & 3) * 2;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&As[r][k]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][k]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&As[r][k + 8]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][k + 8]);
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), k = ks + (ln & 3) * 2;
                unsigned bg[2], bu[2];
                bg[0] = *reinterpret_cast<const unsigned*>(&Bg[c][k]);
                bg[1] = *reinterpret_cast<const unsigned*>(&Bg[c][k + 8]);
                bu[0] = *reinterpret_cast<const unsigned*>(&Bu[c][k]);
                bu[1] = *reinterpret_cast<const unsigned*>(&Bu[c][k + 8]);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    mma_f16(cg[mi][nj], a[mi], bg);
                    mma_f16(cu[mi][nj], a[mi], bu);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int nj = 0; nj < 2; nj++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = wm + mi * 16 + (ln >> 2) + hh * 8, c = wn + nj * 8 + (ln & 3) * 2;
                if (m0 + r >= n) continue;
                const __half2 g2 = *reinterpret_cast<const __half2*>(&cg[mi][nj][hh]);
                const __half2 u2 = *reinterpret_cast<const __half2*>(&cu[mi][nj][hh]);
                float o[2];
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const float gv = fminf(e ? __high2float(g2) : __low2float(g2), CLAMP);
                    const float uv = fminf(fmaxf(e ? __high2float(u2) : __low2float(u2), -CLAMP), CLAMP);
                    o[e] = gv / (1.f + __expf(-gv)) * uv;
                }
                *reinterpret_cast<__half2*>(hb + (size_t)(m0 + r) * ni + n0 + c) = __floats2half2_rn(o[0], o[1]);
            }
}
__global__ void __launch_bounds__(GT, 2) k_moe_gu16(const bf16* xb, const int* rows, int n, XPtr w, int ni, __half* hb) {
    gu16_body(xb, rows, n, w, ni, hb, blockIdx.x, blockIdx.y);
}
// gate/up of many experts in one launch (grid z = job): the resident experts of a layer
__global__ void __launch_bounds__(GT, 2) k_moe_gu16b(const bf16* xb, const int* rows, const GuJob* jobs, int ni, __half* hb) {
    const GuJob& j = jobs[blockIdx.z];
    if ((int)blockIdx.x * QM >= j.n) return;
    gu16_body(xb, rows + j.off, j.n, j.w, ni, hb + (size_t)j.off * ni, blockIdx.x, blockIdx.y);
}
__global__ void __launch_bounds__(GT, 2) k_moe_dn16(const __half* hb, const int* rows, const float* wts, int n, XPtr w,
                                                    int ni, float* part) {
    __shared__ __align__(16) __half As[QM][HK + HPAD];
    __shared__ __align__(16) __half Bs[QN][HK + HPAD];
    const int m0 = blockIdx.x * QM, n0 = blockIdx.y * QN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp >> 2) * 32, wn = (wp & 3) * 16;
    unsigned cc[2][2][2] = {};
    uint4 ra[2];
    unsigned rb[2];
    __half rs[2];
    auto gload = [&](int st) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (HK / 8), c = (i % (HK / 8)) * 8;
            ra[u] = m0 + r < n ? *reinterpret_cast<const uint4*>(hb + (size_t)(m0 + r) * ni + st * HK + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            rb[u] = (unsigned)__ldg(w.dq + (size_t)(w.r0 / 8 + st * (HK / 8) + kk) * H + n0 + c);
            rs[u] = w.ds[(size_t)((w.r0 + st * HK) / 64) * H + n0 + c];
        }
    };
    gload(0);
    for (int st = 0; st < ni / HK; st++) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, r = i / (HK / 8), c = (i % (HK / 8)) * 8;
            *reinterpret_cast<uint4*>(&As[r][c]) = ra[u];
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int i = tid + u * GT, kk = i / QN, c = i % QN;
            *reinterpret_cast<uint4*>(&Bs[c][kk * 8]) = nib2h(rb[u], rs[u]);
        }
        __syncthreads();
        if (st + 1 < ni / HK) gload(st + 1);
#pragma unroll
        for (int ks = 0; ks < HK; ks += 16) {
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r = wm + mi * 16 + (ln >> 2), k = ks + (ln & 3) * 2;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&As[r][k]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][k]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&As[r][k + 8]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][k + 8]);
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), k = ks + (ln & 3) * 2;
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(&Bs[c][k]);
                b[1] = *reinterpret_cast<const unsigned*>(&Bs[c][k + 8]);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) mma_f16(cc[mi][nj], a[mi], b);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = wm + mi * 16 + (ln >> 2) + hh * 8;
            if (m0 + r >= n) continue;
            const float wr = wts[m0 + r];
            float* o = part + (size_t)rows[m0 + r] * H + n0 + wn + (ln & 3) * 2;
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const __half2 v = *reinterpret_cast<const __half2*>(&cc[mi][nj][hh]);
                o[nj * 8] += wr * __low2float(v);
                o[nj * 8 + 1] += wr * __high2float(v);
            }
        }
}
void moe16_gu_batch(const bf16* xb, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni, __half* hb,
                    cudaStream_t st) {
    if (!njobs) return;
    k_moe_gu16b<<<dim3((maxn + QM - 1) / QM, ni / QN, njobs), GT, 0, st>>>(xb, rows, jobs, ni, hb);
}
void moe16_dn(const __half* hb, const int* rows, const float* wts, int n, const XPtr& w, int ni, float* part, cudaStream_t st) {
    if (!n) return;
    k_moe_dn16<<<dim3((n + QM - 1) / QM, H / QN), GT, 0, st>>>(hb, rows, wts, n, w, ni, part);
}
void moe16_expert(const bf16* xb, const int* rows, const float* wts, int n, const XPtr& w, int ni, __half* hb, float* part,
                  cudaStream_t st) {
    if (!n) return;
    const int mb = (n + QM - 1) / QM;
    k_moe_gu16<<<dim3(mb, ni / QN), GT, 0, st>>>(xb, rows, n, w, ni, hb);
    k_moe_dn16<<<dim3(mb, H / QN), GT, 0, st>>>(hb, rows, wts, n, w, ni, part);
}

// ---------------------------------------------------------------- tuned fp16 core: 128-row tiles, 4-stage cp.async
// pipeline (A rows gathered from a fp16 copy of the activations, INT4 weight words and scales copied raw), ldmatrix
// for A with an XOR-swizzled layout, B fragments decoded from the raw words in registers (same values and the same
// ascending k order as k_moe_gu16 / k_moe_dn16: bit-identical results)
__device__ __forceinline__ void cp16(void* sm, const void* gm, bool v) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gm), "r"(v ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// byte b of word w (nibbles k = 2b, 2b + 1) -> half2 (k even low) times the scale, as nib2h
__device__ __forceinline__ unsigned byte2h(unsigned w, int b, __half2 s2) {
    const unsigned x = (w >> (8 * b)) & 0xFFu;
    const unsigned u = (x & 0xFu) | ((x >> 4) << 16) | 0x64006400u;
    __half2 h = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&u), __float2half2_rn(1032.f)), s2);
    return *reinterpret_cast<unsigned*>(&h);
}
constexpr int TM = 64, TK = 32, NST = 4;  // rows per block (128 measured slower: row padding of small experts), k per stage (halves), pipeline stages
// A tile [TM][TK] halves = 4 chunks of 16 B per row, chunk c stored at c ^ ((row >> 1) & 3)
__device__ __forceinline__ int aswz(int r, int c) { return r * TK + ((c ^ ((r >> 1) & 3)) << 3); }

template <int TN>  // columns of gate (and of up) per block
struct GuSm {
    __half A[NST][TM * TK];
    unsigned Bg[NST][TK / 8][TN], Bu[NST][TK / 8][TN];
    __half Sg[NST][TN], Su[NST][TN];
    __half Dg[TN][TK + 8], Du[TN][TK + 8];  // the current stage's weights decoded once (shared by all warps)
};
template <int TN>
__device__ __forceinline__ void gut_body(const __half* x16, const int* rows, int n, const XPtr& w, int ni, __half* hb, int bx,
                                         int by, GuSm<TN>& sm, int* srow) {
    constexpr int WM = 4, WN = 2, MI = TM / WM / 16, NJ = TN / WN / 8;  // 8 warps: 4 x 2, warp tile 32 x (TN / 2)
    const int m0 = bx * TM, n0 = by * TN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp / WN) * (TM / WM), wn = (wp % WN) * (TN / WN);
    for (int i = tid; i < TM; i += 256) srow[i] = m0 + i < n ? rows[m0 + i] : -1;
    __syncthreads();
    auto issue = [&](int stg, int st) {
        const int k0 = st * TK;
        for (int i = tid; i < TM * 4; i += 256) {  // A: 4 chunks per row
            const int r = i >> 2, c = i & 3, row = srow[r];
            cp16(&sm.A[stg][aswz(r, c)], x16 + (size_t)(row >= 0 ? row : 0) * H + k0 + c * 8, row >= 0);
        }
        for (int i = tid; i < 2 * (TK / 8) * (TN / 4); i += 256) {  // B words: 4 packed rows x TN, 4 words per chunk
            const int mtx = i / ((TK / 8) * (TN / 4)), j = i % ((TK / 8) * (TN / 4)), kk = j / (TN / 4), c = (j % (TN / 4)) * 4;
            const int* src = (mtx ? w.uq : w.gq) + (size_t)(st * (TK / 8) + kk) * w.ld + n0 + c;
            cp16(mtx ? &sm.Bu[stg][kk][c] : &sm.Bg[stg][kk][c], src, true);
        }
        for (int i = tid; i < 2 * (TN / 8); i += 256) {  // scales of the stage's 128-group
            const int mtx = i / (TN / 8), c = (i % (TN / 8)) * 8;
            const __half* src = (mtx ? w.us : w.gs) + (size_t)(k0 / 128) * w.ld + n0 + c;
            cp16(mtx ? &sm.Su[stg][c] : &sm.Sg[stg][c], src, true);
        }
    };
    unsigned cg[MI][NJ][2] = {}, cu[MI][NJ][2] = {};
    const int nst = H / TK;
#pragma unroll
    for (int s0 = 0; s0 < NST - 1; s0++) { issue(s0, s0); cp_commit(); }
    for (int st = 0; st < nst; st++) {
        cp_wait<NST - 2>();
        __syncthreads();
        const int stg = st % NST;
        {   // decode the stage's weight words once: 2 matrices x 4 packed rows x TN columns
            for (int i = tid; i < 2 * (TK / 8) * TN; i += 256) {
                const int mtx = i / ((TK / 8) * TN), j = i % ((TK / 8) * TN), kk = j / TN, c = j % TN;
                const unsigned wv = mtx ? sm.Bu[stg][kk][c] : sm.Bg[stg][kk][c];
                const __half sc = mtx ? sm.Su[stg][c] : sm.Sg[stg][c];
                *reinterpret_cast<uint4*>(mtx ? &sm.Du[c][kk * 8] : &sm.Dg[c][kk * 8]) = nib2h(wv, sc);
            }
        }
        __syncthreads();
        if (st + NST - 1 < nst) issue((st + NST - 1) % NST, st + NST - 1);
        cp_commit();
#pragma unroll
        for (int kh = 0; kh < TK / 16; kh++) {
            unsigned a[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int r = wm + mi * 16 + (ln & 15), c = kh * 2 + (ln >> 4);
                ldsm4(a[mi], &sm.A[stg][aswz(r, c)]);
            }
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), k = kh * 16 + (ln & 3) * 2;
                unsigned bg[2], bu[2];
                bg[0] = *reinterpret_cast<const unsigned*>(&sm.Dg[c][k]);
                bg[1] = *reinterpret_cast<const unsigned*>(&sm.Dg[c][k + 8]);
                bu[0] = *reinterpret_cast<const unsigned*>(&sm.Du[c][k]);
                bu[1] = *reinterpret_cast<const unsigned*>(&sm.Du[c][k + 8]);
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    mma_f16(cg[mi][nj], a[mi], bg);
                    mma_f16(cu[mi][nj], a[mi], bu);
                }
            }
        }
        __syncthreads();  // Dg / Du rewritten by the next stage
    }
    cp_wait<0>();
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int nj = 0; nj < NJ; nj++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = wm + mi * 16 + (ln >> 2) + hh * 8, c = wn + nj * 8 + (ln & 3) * 2;
                if (m0 + r >= n) continue;
                const __half2 g2 = *reinterpret_cast<const __half2*>(&cg[mi][nj][hh]);
                const __half2 u2 = *reinterpret_cast<const __half2*>(&cu[mi][nj][hh]);
                float o[2];
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const float gv = fminf(e ? __high2float(g2) : __low2float(g2), CLAMP);
                    const float uv = fminf(fmaxf(e ? __high2float(u2) : __low2float(u2), -CLAMP), CLAMP);
                    o[e] = gv / (1.f + __expf(-gv)) * uv;
                }
                *reinterpret_cast<__half2*>(hb + (size_t)(m0 + r) * ni + n0 + c) = __floats2half2_rn(o[0], o[1]);
            }
}
constexpr int GTN = 64;
__global__ void __launch_bounds__(256) k_moe_gut(const __half* x16, const int* rows, int n, XPtr w, int ni, __half* hb) {
    extern __shared__ __align__(16) char smraw[];
    __shared__ int srow[TM];
    gut_body<GTN>(x16, rows, n, w, ni, hb, blockIdx.x, blockIdx.y, *reinterpret_cast<GuSm<GTN>*>(smraw), srow);
}
__global__ void __launch_bounds__(256) k_moe_gutb(const __half* x16, const int* rows, const GuJob* jobs, int ni, __half* hb) {
    extern __shared__ __align__(16) char smraw[];
    __shared__ int srow[TM];
    const GuJob& j = jobs[blockIdx.z];
    if ((int)blockIdx.x * TM >= j.n) return;
    gut_body<GTN>(x16, rows + j.off, j.n, j.w, ni, hb + (size_t)j.off * ni, blockIdx.x, blockIdx.y,
                  *reinterpret_cast<GuSm<GTN>*>(smraw), srow);
}

// down: tile 128 rows x 128 output columns, 8 warps 2 x 4 of 64 x 32; A = h rows (contiguous), B = down words
constexpr int DTN = 128;
struct DnSm {
    __half A[NST][TM * TK];
    unsigned B[NST][TK / 8][DTN];
    __half S[NST][DTN];
};
__global__ void __launch_bounds__(256) k_moe_dnt(const __half* hb, const int* rows, const float* wts, int n, XPtr w, int ni,
                                                 float* part) {
    extern __shared__ __align__(16) char smraw[];
    DnSm& sm = *reinterpret_cast<DnSm*>(smraw);
    constexpr int WM = 2, WN = 4, MI = TM / WM / 16, NJ = DTN / WN / 8;
    const int m0 = blockIdx.x * TM, n0 = blockIdx.y * DTN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;
    const int wm = (wp / WN) * (TM / WM), wn = (wp % WN) * (DTN / WN);
    auto issue = [&](int stg, int st) {
        const int k0 = st * TK;
        for (int i = tid; i < TM * 4; i += 256) {
            const int r = i >> 2, c = i & 3;
            const bool v = m0 + r < n;
            cp16(&sm.A[stg][aswz(r, c)], hb + (size_t)(v ? m0 + r : 0) * ni + k0 + c * 8, v);
        }
        for (int i = tid; i < (TK / 8) * (DTN / 4); i += 256) {
            const int kk = i / (DTN / 4), c = (i % (DTN / 4)) * 4;
            cp16(&sm.B[stg][kk][c], w.dq + (size_t)(w.r0 / 8 + st * (TK / 8) + kk) * H + n0 + c, true);
        }
        for (int i = tid; i < DTN / 8; i += 256) {
            const int c = i * 8;
            cp16(&sm.S[stg][c], w.ds + (size_t)((w.r0 + k0) / 64) * H + n0 + c, true);
        }
    };
    unsigned cc[MI][NJ][2] = {};
    const int nst = ni / TK;
#pragma unroll
    for (int s0 = 0; s0 < NST - 1; s0++) { if (s0 < nst) issue(s0, s0); cp_commit(); }
    for (int st = 0; st < nst; st++) {
        cp_wait<NST - 2>();
        __syncthreads();
        const int stg = st % NST;
        if (st + NST - 1 < nst) issue((st + NST - 1) % NST, st + NST - 1);
        cp_commit();
#pragma unroll
        for (int kh = 0; kh < TK / 16; kh++) {
            unsigned a[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int r = wm + mi * 16 + (ln & 15), c = kh * 2 + (ln >> 4);
                ldsm4(a[mi], &sm.A[stg][aswz(r, c)]);
            }
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                const int c = wn + nj * 8 + (ln >> 2), t = ln & 3;
                const __half2 s2 = __half2half2(sm.S[stg][c]);
                unsigned b[2] = {byte2h(sm.B[stg][kh * 2][c], t, s2), byte2h(sm.B[stg][kh * 2 + 1][c], t, s2)};
#pragma unroll
                for (int mi = 0; mi < MI; mi++) mma_f16(cc[mi][nj], a[mi], b);
            }
        }
    }
    cp_wait<0>();
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = wm + mi * 16 + (ln >> 2) + hh * 8;
            if (m0 + r >= n) continue;
            const float wr = wts[m0 + r];
            float* o = part + (size_t)rows[m0 + r] * H + n0 + wn + (ln & 3) * 2;
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                const __half2 v = *reinterpret_cast<const __half2*>(&cc[mi][nj][hh]);
                o[nj * 8] += wr * __low2float(v);
                o[nj * 8 + 1] += wr * __high2float(v);
            }
        }
}
static void tuned_attrs() {
    static bool done = false;  // per process; dynamic shared memory sizes (all under 48 KB, attribute set for safety)
    (void)done;
}
void moe16t_expert(const __half* x16, const int* rows, const float* wts, int n, const XPtr& w, int ni, __half* hb, float* part,
                   bool gate_up, cudaStream_t st) {
    if (!n) return;
    tuned_attrs();
    const int mb = (n + TM - 1) / TM;
    if (gate_up) k_moe_gut<<<dim3(mb, ni / GTN), 256, sizeof(GuSm<GTN>), st>>>(x16, rows, n, w, ni, hb);
    k_moe_dnt<<<dim3(mb, H / DTN), 256, sizeof(DnSm), st>>>(hb, rows, wts, n, w, ni, part);
}
void moe16t_dn(const __half* hb, const int* rows, const float* wts, int n, const XPtr& w, int ni, float* part, cudaStream_t st) {
    if (!n) return;
    k_moe_dnt<<<dim3((n + TM - 1) / TM, H / DTN), 256, sizeof(DnSm), st>>>(hb, rows, wts, n, w, ni, part);
}
void moe16t_gu_batch(const __half* x16, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni, __half* hb,
                     cudaStream_t st) {
    if (!njobs) return;
    k_moe_gutb<<<dim3((maxn + TM - 1) / TM, ni / GTN, njobs), 256, sizeof(GuSm<GTN>), st>>>(x16, rows, jobs, ni, hb);
}

void moe8_gu(const int8_t* xq, const float* xs, const int* rows, int n, const XPtr& w, int ni, int8_t* hq, float* hs,
             cudaStream_t st) {
    if (!n) return;
    k_moe_gu8<<<dim3((n + QM - 1) / QM, ni / QN), GT, 0, st>>>(xq, xs, rows, n, w, ni, hq, hs);
}
void moe8_dn(const int8_t* hq, const float* hs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
             float* part, cudaStream_t st) {
    if (!n) return;
    k_moe_dn8<<<dim3((n + QM - 1) / QM, H / QN), GT, 0, st>>>(hq, hs, rows, wts, n, w, ni, part);
}
void moe8_expert(const int8_t* xq, const float* xs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
                 int8_t* hq, float* hs, float* part, cudaStream_t st) {
    if (!n) return;
    const int mb = (n + QM - 1) / QM;
    k_moe_gu8<<<dim3(mb, ni / QN), GT, 0, st>>>(xq, xs, rows, n, w, ni, hq, hs);
    k_moe_dn8<<<dim3(mb, H / QN), GT, 0, st>>>(hq, hs, rows, wts, n, w, ni, part);
}

// ---------------------------------------------------------------- round 23: pipelined W4A8 expert kernels
// Same int8 products and the same fp32 expressions as k_moe_gu8b / k_moe_dn8, so the outputs are bit-identical:
// - k order inside each 32-wide mma step is permuted (logical k 4j..4j+3 -> physical 8j..8j+3, 16+4j.. -> 8j+4..), so
//   a thread's B fragment is exactly one INT4 word (8 consecutive k of one column) decoded in registers (nib2s8) and its
//   A fragment one 8-byte load per row; the int32 sum of a step is the same integer in any order.
// - multi-stage cp.async pipeline (raw INT4 words, int8 rows gathered by the row list, scales), one barrier per K-group.
// - the per-K-group rescale (float)acc * sx * sw added in ascending group order, glu and the h quantisation exactly as
//   before; the down epilogue adds wr * f into part per expert in launch order (deterministic).
__device__ __forceinline__ void cpa16(void* s, const void* g, bool p) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(p ? 16 : 0));
}
__device__ __forceinline__ void cpa4(void* s, const void* g, bool p) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sa), "l"(g), "r"(p ? 4 : 0));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// gate/up: tile BM rows x 64 columns of gate and of up, 8 warps 2 (m) x 4 (n) of BM/2 x 16 (+16); K-group 128 per stage
template <int BM, int ST> struct GuN {
    static constexpr int MI = BM / 32, BW = 64 + 8;  // m16 tiles per warp, B row stride (words)
    static constexpr int A_B = BM * 128, B_B = 16 * BW * 4, X_B = BM * 4, S_B = 128 * 2;
    static constexpr int STAGE = A_B + 2 * B_B + X_B + S_B;
    static constexpr int SMEM = ST * STAGE + BM * 4;
};
template <int BM, int ST>
__global__ void __launch_bounds__(256, BM == 128 ? 1 : 2) k_moe_gu8n(const int8_t* xq, const float* xs, const int* rows_all,
                                                                     const GuJob* jobs, int ni, int8_t* hq_all, float* hs_all) {
    using C = GuN<BM, ST>;
    constexpr int MI = C::MI, BW = C::BW;
    const GuJob& jb = jobs[blockIdx.z];
    const int n = jb.n;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * 64, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;  // n fastest: A tile shared in L2
    if (m0 >= n) return;
    const XPtr w = jb.w;
    const int* rows = rows_all + jb.off;
    int8_t* hq = hq_all + (size_t)jb.off * ni;
    float* hs = hs_all + (size_t)jb.off * (ni / 64);
    extern __shared__ __align__(16) unsigned char smem[];
    int* srow = reinterpret_cast<int*>(smem + ST * C::STAGE);
    const int wm = (wp >> 2) * (BM / 2), wn = (wp & 3) * 16, j = ln & 3;
    for (int r = tid; r < BM; r += 256) srow[r] = m0 + r < n ? rows[m0 + r] : -1;
    __syncthreads();
    auto stage = [&](int s) { return smem + s * C::STAGE; };
    auto load = [&](int g, int s) {
        unsigned char* b = stage(s);
#pragma unroll
        for (int u = 0; u < BM * 8 / 256; u++) {
            const int i = tid + u * 256, r = i >> 3, c = i & 7, row = srow[r];
            cpa16(b + r * 128 + ((c ^ ((r & 3) << 1)) << 4), xq + (size_t)(row >= 0 ? row : 0) * H + g * 128 + c * 16, row >= 0);
        }
        {
            const int kk = tid >> 4, c = tid & 15;
            const size_t wo = (size_t)(g * 16 + kk) * w.ld + n0 + c * 4;
            cpa16(b + C::A_B + (kk * BW + c * 4) * 4, w.gq + wo, true);
            cpa16(b + C::A_B + C::B_B + (kk * BW + c * 4) * 4, w.uq + wo, true);
        }
        for (int r = tid; r < BM; r += 256) {
            const int row = srow[r];
            cpa4(b + C::A_B + 2 * C::B_B + r * 4, xs + (size_t)(row >= 0 ? row : 0) * (H / 128) + g, row >= 0);
        }
        if (tid < 16) {
            const __half* src = (tid < 8 ? w.gs : w.us) + (size_t)g * w.ld + n0 + (tid & 7) * 8;
            cpa16(b + C::A_B + 2 * C::B_B + C::X_B + tid * 16, src, true);
        }
    };
    float fg[MI][2][4] = {}, fu[MI][2][4] = {};
    constexpr int NG = H / 128;
#pragma unroll
    for (int s = 0; s < ST - 1; s++) { load(s, s); cpa_commit(); }
    for (int g = 0; g < NG; g++) {
        cpa_wait<ST - 2>();
        __syncthreads();
        if (g + ST - 1 < NG) load(g + ST - 1, (g + ST - 1) % ST);
        cpa_commit();
        const unsigned char* b = stage(g % ST);
        const unsigned* Bg = reinterpret_cast<const unsigned*>(b + C::A_B);
        const unsigned* Bu = reinterpret_cast<const unsigned*>(b + C::A_B + C::B_B);
        const float* sx = reinterpret_cast<const float*>(b + C::A_B + 2 * C::B_B);
        const __half* sgs = reinterpret_cast<const __half*>(b + C::A_B + 2 * C::B_B + C::X_B);
        int ag[MI][2][4] = {}, au[MI][2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            unsigned a[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int r = wm + mi * 16 + (ln >> 2);
                const int o = (((2 * ks + (j >> 1)) ^ ((r & 3) << 1)) << 4) + (j & 1) * 8;
                const uint2 v0 = *reinterpret_cast<const uint2*>(b + r * 128 + o);
                const uint2 v1 = *reinterpret_cast<const uint2*>(b + (r + 8) * 128 + o);
                a[mi][0] = v0.x; a[mi][2] = v0.y; a[mi][1] = v1.x; a[mi][3] = v1.y;
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                const int c = wn + nj * 8 + (ln >> 2);
                unsigned bg[2], bu[2];
                nib2s8(Bg[(4 * ks + j) * BW + c], bg[0], bg[1]);
                nib2s8(Bu[(4 * ks + j) * BW + c], bu[0], bu[1]);
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    mma_s8(ag[mi][nj], a[mi], bg);
                    mma_s8(au[mi][nj], a[mi], bu);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < 2; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                    const float swg = __half2float(sgs[c]), swu = __half2float(sgs[64 + c]);
                    fg[mi][nj][e] += (float)ag[mi][nj][e] * sx[r] * swg;
                    fu[mi][nj][e] += (float)au[mi][nj][e] * sx[r] * swu;
                }
    }
    cpa_wait<0>();
    __syncthreads();
    float* hsm = reinterpret_cast<float*>(smem);  // [BM][64]
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int nj = 0; nj < 2; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                const float gv = fminf(fg[mi][nj][e], CLAMP), uv = fminf(fmaxf(fu[mi][nj][e], -CLAMP), CLAMP);
                hsm[r * 64 + c] = gv / (1.f + __expf(-gv)) * uv;
            }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < BM / 64; pass++) {
        const int r = (tid >> 2) + pass * 64, q4 = tid & 3;
        float m = 0.f;
        for (int c = 0; c < 16; c++) m = fmaxf(m, fabsf(hsm[r * 64 + q4 * 16 + c]));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
        if (m0 + r < n) {
            unsigned pk[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                unsigned v = 0;
#pragma unroll
                for (int t = 0; t < 4; t++)
                    v |= (unsigned)(unsigned char)(int8_t)__float2int_rn(hsm[r * 64 + q4 * 16 + q * 4 + t] * inv) << (8 * t);
                pk[q] = v;
            }
            *reinterpret_cast<uint4*>(hq + (size_t)(m0 + r) * ni + n0 + q4 * 16) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            if (q4 == 0) hs[(size_t)(m0 + r) * (ni / 64) + n0 / 64] = sc;
        }
    }
}

// down: tile 64 rows x BN output columns, 8 warps 2 (m) x 4 (n) of 32 x BN/4, K-group 64 per stage; the row scales of
// all groups and the column scales of the tile loaded once (BN 128; a variant that copied the tile's part values into
// shared memory at the start was slower: the part reads queue ahead of the weight loads)
constexpr int DN_ST = 4, DN_MAXG = 16;
template <int BN> struct DnN {
    static constexpr int BM = 64, BW = BN + 8, NJ = BN / 32;
    static constexpr int A_B = BM * 64, B_B = 8 * BW * 4, STAGE = A_B + B_B;
    static constexpr int SMEM = DN_ST * STAGE + BM * DN_MAXG * 4 + DN_MAXG * BN * 2;
};
template <int BN>
__global__ void __launch_bounds__(256, 2) k_moe_dn8n(const int8_t* hq, const float* hs, const int* rows, const float* wts,
                                                     int n, XPtr w, int ni, float* part) {
    using C = DnN<BN>;
    constexpr int BW = C::BW, NJ = C::NJ;
    const int m0 = blockIdx.y * C::BM, n0 = blockIdx.x * BN, tid = threadIdx.x, wp = tid >> 5, ln = tid & 31;  // n fastest
    const int wm = (wp >> 2) * 32, wn = (wp & 3) * (BN / 4), j = ln & 3, NG = ni / 64;
    extern __shared__ __align__(16) unsigned char smem[];
    float* sxa = reinterpret_cast<float*>(smem + DN_ST * C::STAGE);          // [BM][DN_MAXG]
    __half* sws = reinterpret_cast<__half*>(sxa + C::BM * DN_MAXG);           // [DN_MAXG][BN]
    for (int i = tid; i < C::BM * NG; i += 256) {
        const int r = i / NG, g = i % NG;
        cpa4(sxa + r * DN_MAXG + g, hs + (size_t)(m0 + r < n ? m0 + r : 0) * NG + g, m0 + r < n);
    }
    for (int i = tid; i < NG * (BN / 8); i += 256) {
        const int g = i / (BN / 8), c = (i % (BN / 8)) * 8;
        cpa16(sws + g * BN + c, w.ds + (size_t)((w.r0 + g * 64) / 64) * H + n0 + c, true);
    }
    auto stage = [&](int s) { return smem + s * C::STAGE; };
    auto load = [&](int g, int s) {
        unsigned char* b = stage(s);
        {
            const int r = tid >> 2, c = tid & 3;
            const bool v = m0 + r < n;
            cpa16(b + r * 64 + ((c ^ (((r >> 1) & 1) << 1)) << 4), hq + (size_t)(v ? m0 + r : 0) * ni + g * 64 + c * 16, v);
        }
        if (tid < 8 * (BN / 4)) {
            const int kk = tid / (BN / 4), c = tid % (BN / 4);
            cpa16(b + C::A_B + (kk * BW + c * 4) * 4, w.dq + (size_t)(w.r0 / 8 + g * 8 + kk) * H + n0 + c * 4, true);
        }
    };
    float f[2][NJ][4] = {};
#pragma unroll
    for (int s = 0; s < DN_ST - 1; s++) { if (s < NG) load(s, s); cpa_commit(); }  // group 0 also holds the scales
    for (int g = 0; g < NG; g++) {
        cpa_wait<DN_ST - 2>();
        __syncthreads();
        if (g + DN_ST - 1 < NG) load(g + DN_ST - 1, (g + DN_ST - 1) % DN_ST);
        cpa_commit();
        const unsigned char* b = stage(g % DN_ST);
        const unsigned* Bs = reinterpret_cast<const unsigned*>(b + C::A_B);
        int acc[2][NJ][4] = {};
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r = wm + mi * 16 + (ln >> 2);
                const int o = (((2 * ks + (j >> 1)) ^ (((r >> 1) & 1) << 1)) << 4) + (j & 1) * 8;
                const uint2 v0 = *reinterpret_cast<const uint2*>(b + r * 64 + o);
                const uint2 v1 = *reinterpret_cast<const uint2*>(b + (r + 8) * 64 + o);
                a[mi][0] = v0.x; a[mi][2] = v0.y; a[mi][1] = v1.x; a[mi][3] = v1.y;
            }
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                unsigned bb[2];
                nib2s8(Bs[(4 * ks + j) * BW + wn + nj * 8 + (ln >> 2)], bb[0], bb[1]);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) mma_s8(acc[mi][nj], a[mi], bb);
            }
        }
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8, c = wn + nj * 8 + (ln & 3) * 2 + (e & 1);
                    f[mi][nj][e] += (float)acc[mi][nj][e] * sxa[r * DN_MAXG + g] * __half2float(sws[g * BN + c]);
                }
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int r = wm + mi * 16 + (ln >> 2) + (e >> 1) * 8;
            if (m0 + r >= n) continue;
            const float wr = wts[m0 + r];
            const int c0 = wn + (ln & 3) * 2 + (e & 1);
            float* o = part + (size_t)rows[m0 + r] * H + n0 + c0;
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) o[nj * 8] += wr * f[mi][nj][e];
        }
}

static bool al16(const void* p) { return ((uintptr_t)p & 15) == 0; }
// dynamic shared memory above 48 KB: the attribute belongs to each device's context, so it is set once per device
template <class K> static void smem_attr(K k, int bytes, std::atomic<unsigned>& done) {
    if (bytes <= 48 * 1024) return;  // today's kernels (36 / 42 KB) need no attribute
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    const unsigned bit = dev < 32 ? 1u << dev : 0u;  // review r23 L1: devices >= 32 set it on every call
    if (!bit || !(done.load(std::memory_order_acquire) & bit)) {
        CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        done.fetch_or(bit, std::memory_order_acq_rel);
    }
}
// review r23-L4: read like the other flags (env_int: empty = unset); the engine calls it once at load (one reader, the
// mode is logged there), so the static is never first initialised from three worker threads
int moe8n_mode() {
    static const int m = env_int("GLM_PF_MOEN", 1);
    return m;
}
void moe8n_gu_batch(const int8_t* xq, const float* xs, const int* rows, const GuJob* jobs, int njobs, int maxn, int ni,
                    int8_t* hq, float* hs, cudaStream_t st) {
    if (!njobs || !maxn) return;
    // review r23-L2: the jobs' weights are device-side (d_gujobs), so their pointers are not checked here. Contract of the
    // callers: every job's gq / uq / gs / us lie in 256 B-aligned slab or ring slices with ld == ni (a multiple of 64)
    if (!moe8n_mode() || (ni & 63) || !al16(xq)) { moe8_gu_batch(xq, xs, rows, jobs, njobs, maxn, ni, hq, hs, st); return; }
    using C = GuN<64, 2>;  // 64-row tiles, 2 CTAs per SM (128-row tiles at 1 CTA per SM measured 2 % slower)
    static std::atomic<unsigned> d{0};
    smem_attr(k_moe_gu8n<64, 2>, C::SMEM, d);
    k_moe_gu8n<64, 2><<<dim3(ni / 64, (maxn + 63) / 64, njobs), 256, C::SMEM, st>>>(xq, xs, rows, jobs, ni, hq, hs);
}
void moe8n_dn(const int8_t* hq, const float* hs, const int* rows, const float* wts, int n, const XPtr& w, int ni,
              float* part, cudaStream_t st) {
    if (!n) return;
    // review r23-L3: a host-resident expert takes the old form too (none reaches this path today: the ring replaces cold
    // experts; the down kernel reads its weights with the H stride, so w.ld needs no check here)
    if (!moe8n_mode() || w.host || (ni & 63) || ni / 64 > DN_MAXG || (w.r0 & 63) || !al16(hq) || !al16(w.dq) ||
        !al16(w.ds)) {
        moe8_dn(hq, hs, rows, wts, n, w, ni, part, st);
        return;
    }
    static std::atomic<unsigned> d{0};
    smem_attr(k_moe_dn8n<128>, DnN<128>::SMEM, d);
    k_moe_dn8n<128><<<dim3(H / 128, (n + 63) / 64), 256, DnN<128>::SMEM, st>>>(hq, hs, rows, wts, n, w, ni, part);
}

}  // namespace glm
